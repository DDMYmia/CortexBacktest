"""Order management for the event-driven broker."""

from __future__ import annotations

from dataclasses import replace
from heapq import heappop, heappush
from datetime import datetime
from collections import defaultdict
from typing import Callable

from ..contracts import QUANTITY_TOLERANCE, Bar, Order, OrderRecord, OrderStatus, OrderType, Side, TimeInForce


def _order_can_expire(order: Order) -> bool:
    return order.expires_at is not None or order.time_in_force != TimeInForce.GTC


# Per-symbol "could anything here trigger on this bar?" envelope, held as a
# plain list for cheap indexed access in the event loop. Each slot keeps the
# *most easily triggered* resting price in one direction, so the question is
# four comparisons rather than a walk over the symbol's open orders:
#
#   a limit buy   fills when bar.low  <= price  -> easiest is the highest
#   a limit sell  fills when bar.high >= price  -> easiest is the lowest
#   a stop buy    fires when bar.high >= price  -> easiest is the lowest
#   a stop sell   fires when bar.low  <= price  -> easiest is the highest
#
# `_E_ALWAYS` counts orders no price bound can exclude (market, trailing
# stop, or a limit/stop whose price is None): while any is resting the
# envelope cannot rule the symbol out.
_E_MAX_LIMIT_BUY = 0
_E_MIN_LIMIT_SELL = 1
_E_MIN_STOP_BUY = 2
_E_MAX_STOP_SELL = 3
_E_ALWAYS = 4
_E_STALE = 5


# The value a slot holds when nothing rests in it: an extreme so permissive it
# can never make `symbol_can_trigger` say no on its own.
_EMPTY_SLOT = (-float("inf"), float("inf"), float("inf"), -float("inf"))
# heapq is a min-heap, so the two "highest wins" slots order on the negated
# price. Keeping the sign here means the heaps and the scalars agree by
# construction rather than by remembering to flip at each use.
_SLOT_SIGN = (-1.0, 1.0, 1.0, -1.0)


def _new_envelope() -> list[float]:
    return [*_EMPTY_SLOT, 0.0, 0.0]


def _new_envelope_heaps() -> tuple[list, list, list, list]:
    return ([], [], [], [])


class OrderManager:
    """Own open orders, IDs, and lifecycle records."""

    def __init__(
        self,
        *,
        retain_records: bool = True,
        record_sink: Callable[[OrderRecord], None] | None = None,
    ) -> None:
        # Insertion-ordered and authoritative: `open_orders` is this dict's
        # values, so there is no second flat list of ids to keep in step.
        self._open_orders_by_id: dict[str, Order] = {}
        # Exact secondary indexes. An id leaves these in the same call that
        # removes the order, so a lookup walks live entries only — it never
        # skips dead ids and never needs a `seen` set to dedupe. dict-as-
        # ordered-set buys O(1) removal while keeping the insertion order
        # price-time priority depends on; the previous append-only lists made
        # removal O(1) too, but pushed the cost onto every later read.
        self._open_order_ids_by_symbol: dict[str, dict[str, None]] = defaultdict(dict)
        # OCO groups are not scoped to one symbol (a bracket's take-profit and
        # stop-loss share a symbol in every example in this codebase, but the
        # contract does not require it — test_matching_path_equivalence.py's
        # fuzz cases deliberately assign the same oco_group across different
        # symbols). cancel_oco_siblings needs "every open order in this
        # group", which a symbol index cannot answer.
        self._open_order_ids_by_oco_group: dict[str, dict[str, None]] = defaultdict(dict)
        # symbol -> envelope (see _new_envelope). Widened on add, left
        # deliberately stale on remove — a stale envelope is *permissive*, so
        # the gate can only ever be over-cautious, never skip a fillable
        # order — and rebuilt once staleness outgrows the live orders.
        self._trigger_envelope_by_symbol: dict[str, list[float]] = {}
        # One lazy heap per envelope slot per symbol. They exist only so that
        # losing an extreme costs a few pops instead of a rescan of every order
        # resting on the symbol; the read path still reads the scalars.
        self._envelope_heaps_by_symbol: dict[str, tuple[list, list, list, list]] = {}
        self._triggered_stop_limits: set[str] = set()
        self._expirable_open_order_count = 0
        self._expirable_order_ids: dict[str, None] = {}
        # Maintained incrementally instead of len(self._open_orders_by_id) so the
        # per-bar "any open orders?" check is a plain attribute read, not a
        # property call, in the hot event loop.
        self.open_order_count = 0
        self._held_orders_by_parent: dict[str, list[Order]] = defaultdict(list)
        self.retain_records = retain_records
        # Only accumulate pending records when something actually consumes them
        # (a notify_order hook). Without a consumer they are drained-and-dropped
        # every bar, so skipping the append avoids per-bar work in summary runs.
        self.track_pending_records = True
        self.record_sink = record_sink
        self.order_records: list[OrderRecord] = []
        self.pending_records: list[OrderRecord] = []
        self.order_record_count = 0
        self.submitted_order_count = 0
        self._next_order_id = 1

    @property
    def open_orders(self) -> list[Order]:
        return list(self._open_orders_by_id.values())

    @open_orders.setter
    def open_orders(self, orders: list[Order]) -> None:
        # Held child orders are not open orders: replacing the open set must not
        # discard children still waiting on a parent fill.
        self._open_orders_by_id = {}
        self._open_order_ids_by_symbol = defaultdict(dict)
        self._open_order_ids_by_oco_group = defaultdict(dict)
        self._trigger_envelope_by_symbol = {}
        self._envelope_heaps_by_symbol = {}
        self._expirable_open_order_count = 0
        self._expirable_order_ids = {}
        self.open_order_count = 0
        for order in orders:
            self.add_open_order(order)

    def is_open(self, order_id: str | None) -> bool:
        return order_id is not None and order_id in self._open_orders_by_id

    def prepare_order(self, order: Order) -> Order:
        symbol = order.symbol.upper()
        if order.id is not None and order.symbol == symbol:
            return order
        order_id = order.id or f"O{self._next_order_id:08d}"
        if order.id is None:
            self._next_order_id += 1
        # Direct construction instead of dataclasses.replace: this runs for
        # every submitted order and replace() is several times slower.
        return Order(
            symbol=symbol,
            timestamp=order.timestamp,
            side=order.side,
            quantity=order.quantity,
            order_type=order.order_type,
            limit_price=order.limit_price,
            stop_price=order.stop_price,
            trail_amount=order.trail_amount,
            trail_percent=order.trail_percent,
            tag=order.tag,
            id=order_id,
            expires_at=order.expires_at,
            oco_group=order.oco_group,
            parent_id=order.parent_id,
            time_in_force=order.time_in_force,
        )

    def reject(self, order: Order, reason: str) -> OrderRecord:
        return self.record(order, OrderStatus.REJECTED, reason)

    def submit(self, order: Order) -> OrderRecord:
        record = self.record(order, OrderStatus.SUBMITTED)
        if order.parent_id is not None and order.parent_id in self._open_orders_by_id:
            self._held_orders_by_parent[order.parent_id].append(order)
        else:
            self.add_open_order(order)
        self._note(order, OrderStatus.ACCEPTED)
        return record

    def add_open_order(self, order: Order) -> None:
        if order.id is None:
            raise ValueError("open orders require an id")
        if order.id in self._open_orders_by_id:
            self.remove_open_order(order.id)
        self._open_orders_by_id[order.id] = order
        self.open_order_count += 1
        self._open_order_ids_by_symbol[order.symbol.upper()][order.id] = None
        self._widen_trigger_envelope(order)
        if order.oco_group is not None:
            self._open_order_ids_by_oco_group[order.oco_group][order.id] = None
        if _order_can_expire(order):
            self._expirable_open_order_count += 1
            self._expirable_order_ids[order.id] = None

    def replace_open_order(self, old_order_id: str, order: Order) -> None:
        self.remove_open_order(old_order_id)
        self.add_open_order(order)

    def update_open_order(self, order: Order) -> None:
        """Overwrite an open order in place, keeping its queue position.

        Used for amendments that must not lose price-time priority: trailing-stop
        recalculation and partial-fill remainders. A no-op when the id is not
        open, so a cancelled order can never be resurrected through this path.
        """
        if order.id is None or order.id not in self._open_orders_by_id:
            return
        previous = self._open_orders_by_id[order.id]
        self._open_orders_by_id[order.id] = order
        if _order_can_expire(previous) != _order_can_expire(order):
            self._expirable_open_order_count += 1 if _order_can_expire(order) else -1
            if _order_can_expire(order):
                # An amendment preserves queue priority. Rebuild only on this
                # rare membership transition, so expiration records retain
                # original submission order rather than amendment order.
                self._expirable_order_ids = {key: None for key, value in self._open_orders_by_id.items()
                                             if _order_can_expire(value)}
            else:
                self._expirable_order_ids.pop(order.id, None)
        # A trailing stop ratchets through this path, so its trigger price can
        # move. Release the old contribution before widening for the new one,
        # or an always-candidate order would be counted twice and its symbol
        # could never be gated again.
        self._release_trigger_envelope(previous)
        self._widen_trigger_envelope(order)

    def get_order(self, order_id: str) -> Order | None:
        order = self._open_orders_by_id.get(order_id)
        if order is not None:
            return order
        return self._held_order(order_id)

    def replace_order(self, order_id: str, order: Order, *, reason: str = "replaced") -> OrderRecord | None:
        if order.id != order_id:
            order = replace(order, id=order_id)
        if order_id in self._open_orders_by_id:
            self.replace_open_order(order_id, order)
            return self.record(order, OrderStatus.REPLACED, reason)
        if self._replace_held_order(order_id, order):
            return self.record(order, OrderStatus.REPLACED, reason)
        return None

    def remove_open_order(self, order_id: str) -> Order | None:
        order = self._open_orders_by_id.pop(order_id, None)
        if order is None:
            return None
        self.open_order_count -= 1
        symbol_index = self._open_order_ids_by_symbol.get(order.symbol.upper())
        if symbol_index is not None:
            symbol_index.pop(order_id, None)
        if order.oco_group is not None:
            group_index = self._open_order_ids_by_oco_group.get(order.oco_group)
            if group_index is not None:
                group_index.pop(order_id, None)
        self._release_trigger_envelope(order)
        self._triggered_stop_limits.discard(order_id)
        if _order_can_expire(order):
            self._expirable_open_order_count -= 1
            self._expirable_order_ids.pop(order_id, None)
        return order

    def effective_order_type(self, order: Order) -> OrderType:
        if order.order_type is OrderType.STOP_LIMIT and order.id in self._triggered_stop_limits:
            return OrderType.LIMIT
        return order.order_type

    def arm_stop_limit(self, order: Order) -> None:
        if order.id is not None and order.id not in self._triggered_stop_limits:
            self._triggered_stop_limits.add(order.id)
            self._rebuild_trigger_envelope(order.symbol)

    def _envelope_slot(self, order: Order) -> tuple[int, float] | None:
        """Which envelope extreme this order pins, and at what price.

        None means no price bound can exclude it -- a market order, a trailing
        stop, or a limit/stop whose price is not set -- so the symbol stays
        ungateable while it rests.
        """
        order_type = self.effective_order_type(order)
        if order_type is OrderType.LIMIT:
            price = order.limit_price
            if price is not None:
                return (_E_MAX_LIMIT_BUY if order.side is Side.BUY else _E_MIN_LIMIT_SELL, price)
        elif order_type is OrderType.STOP or order_type is OrderType.STOP_LIMIT:
            price = order.stop_price
            if price is not None:
                return (_E_MIN_STOP_BUY if order.side is Side.BUY else _E_MAX_STOP_SELL, price)
        return None

    def _widen_trigger_envelope(self, order: Order) -> None:
        symbol = order.symbol.upper()
        envelope = self._trigger_envelope_by_symbol.get(symbol)
        if envelope is None:
            envelope = self._trigger_envelope_by_symbol[symbol] = _new_envelope()
        slot_price = self._envelope_slot(order)
        if slot_price is None:
            envelope[_E_ALWAYS] += 1
            return
        slot, price = slot_price
        signed = _SLOT_SIGN[slot] * price
        if signed < _SLOT_SIGN[slot] * envelope[slot]:
            envelope[slot] = price
        heaps = self._envelope_heaps_by_symbol.get(symbol)
        if heaps is None:
            heaps = self._envelope_heaps_by_symbol[symbol] = _new_envelope_heaps()
        heappush(heaps[slot], (signed, price, order.id))

    def _release_trigger_envelope(self, order: Order) -> None:
        symbol = order.symbol.upper()
        envelope = self._trigger_envelope_by_symbol.get(symbol)
        if envelope is None:
            return
        slot_price = self._envelope_slot(order)
        if slot_price is None:
            if envelope[_E_ALWAYS] > 0:
                envelope[_E_ALWAYS] -= 1
            return
        slot, price = slot_price
        # An envelope wider than reality is still correct — it only costs a
        # skip that could have happened. But leaving it wide is what makes the
        # gate useless in practice: the departing order is usually the one
        # pinning an extreme, and a permissive extreme never recovers. So the
        # extreme is recovered exactly when its holder leaves.
        #
        # Recovery pops the slot's heap rather than rescanning every order on
        # the symbol, because that rescan is O(open orders) and open orders
        # grow through a run: measured at 1,000 symbols, throughput fell from
        # 398k to 191k obs/s as the book grew from 9k to 122k resting orders,
        # and the rescan was the only term that grew with it.
        if price == envelope[slot]:
            self._recover_envelope_slot(symbol, envelope, slot)
            return
        envelope[_E_STALE] += 1
        if envelope[_E_STALE] > len(self._open_order_ids_by_symbol.get(symbol, ())):
            self._rebuild_trigger_envelope(symbol)

    def _recover_envelope_slot(self, symbol: str, envelope: list[float], slot: int) -> None:
        """Set one slot to the next surviving extreme, discarding dead entries.

        Entries are never removed on order exit, so the top may name an order
        that has left or been replaced at another price. Each entry is pushed
        once and popped at most once, which is what makes losing an extreme
        amortized O(log n) rather than a full rescan.
        """
        heaps = self._envelope_heaps_by_symbol.get(symbol)
        if heaps is None:
            envelope[slot] = _EMPTY_SLOT[slot]
            return
        heap = heaps[slot]
        by_id = self._open_orders_by_id
        while heap:
            _, price, order_id = heap[0]
            live = by_id.get(order_id)
            if live is not None:
                current = self._envelope_slot(live)
                # The order must still pin *this* slot at *this* price: a
                # trailing stop that ratcheted, or a stop-limit that armed into
                # a limit, leaves an entry that no longer describes it.
                if current is not None and current[0] == slot and current[1] == price:
                    envelope[slot] = price
                    return
            heappop(heap)
        envelope[slot] = _EMPTY_SLOT[slot]

    def _rebuild_trigger_envelope(self, symbol: str) -> None:
        self._trigger_envelope_by_symbol[symbol] = _new_envelope()
        self._envelope_heaps_by_symbol[symbol] = _new_envelope_heaps()
        for order in self.open_orders_for_symbol(symbol):
            self._widen_trigger_envelope(order)

    def symbol_can_trigger(self, symbol: str, bar_low: float, bar_high: float) -> bool:
        """Can any order resting on `symbol` possibly fill within this bar's range?

        Only ever over-reports. A False means every resting order's trigger is
        strictly outside [bar_low, bar_high], which is exactly the condition
        under which the matching loop would have skipped every one of them —
        so callers may skip the loop entirely, but only when they would have
        applied the same price prefilter (i.e. no custom matching model).
        """
        envelope = self._trigger_envelope_by_symbol.get(symbol)
        if envelope is None:
            return False
        if envelope[_E_ALWAYS]:
            return True
        return (
            bar_low <= envelope[_E_MAX_LIMIT_BUY]
            or bar_high >= envelope[_E_MIN_LIMIT_SELL]
            or bar_high >= envelope[_E_MIN_STOP_BUY]
            or bar_low <= envelope[_E_MAX_STOP_SELL]
        )

    def open_orders_for_symbol(self, symbol: str) -> list[Order]:
        by_id = self._open_orders_by_id
        return [by_id[order_id] for order_id in self._open_order_ids_by_symbol.get(symbol.upper(), ())]

    def iter_open_orders_for_symbol(self, symbol: str):
        return iter(self.open_orders_for_symbol(symbol))

    def open_orders_for_oco_group(self, oco_group: str) -> list[Order]:
        by_id = self._open_orders_by_id
        return [by_id[order_id] for order_id in self._open_order_ids_by_oco_group.get(oco_group, ())]

    def record(
        self,
        order: Order,
        status: OrderStatus,
        reason: str | None = None,
    ) -> OrderRecord:
        record = OrderRecord(order, status, reason)
        self.order_record_count += 1
        if status == OrderStatus.SUBMITTED:
            self.submitted_order_count += 1
        if self.retain_records:
            self.order_records.append(record)
        elif self.track_pending_records:
            self.pending_records.append(record)
        if self.record_sink is not None:
            self.record_sink(record)
        return record

    def records_observed(self) -> bool:
        """Whether anything will read an `OrderRecord` that gets built."""
        return self.retain_records or self.track_pending_records or self.record_sink is not None

    def _note(self, order: Order, status: OrderStatus, reason: str | None = None) -> None:
        """Record a transition whose `OrderRecord` no caller reads.

        The fill path logs two or three transitions per order (accepted, then
        partially-filled and/or filled) and discards every one of the returned
        records. When nothing retains them — no record list, no pending queue
        for a notifier, no sink — building the object is pure waste, and
        `OrderRecord`/`Order` are frozen dataclasses at roughly 6x the
        construction cost of plain ones. Only the counters have to move.
        """
        if self.records_observed():
            self.record(order, status, reason)
            return
        self.order_record_count += 1
        if status == OrderStatus.SUBMITTED:
            self.submitted_order_count += 1

    def drain_pending_records(self) -> list[OrderRecord]:
        records = self.pending_records
        self.pending_records = []
        return records

    def cancel_order(self, order_id: str, *, reason: str = "cancelled") -> OrderRecord | None:
        order = self.remove_open_order(order_id)
        if order is None:
            order = self._remove_held_order(order_id)
        if order is None:
            return None
        self._cancel_children(order_id, reason=reason)
        return self.record(order, OrderStatus.CANCELLED, reason)

    def cancel_symbol(self, symbol: str, *, reason: str = "cancelled") -> list[OrderRecord]:
        symbol = symbol.upper()
        cancelled = []
        for order in self.open_orders_for_symbol(symbol):
            self.remove_open_order(order.id or "")
            self._cancel_children(order.id or "", reason=reason)
            cancelled.append(self.record(order, OrderStatus.CANCELLED, reason))
        for order in self._held_orders_for_symbol(symbol):
            self._remove_held_order(order.id or "")
            cancelled.append(self.record(order, OrderStatus.CANCELLED, reason))
        return cancelled

    def close_order_after_fill(self, order: Order, fill_quantity: float) -> Order | None:
        """Record fill status and return a remaining open order, if any."""
        if fill_quantity <= 0:
            raise ValueError("fill quantity must be positive")
        if fill_quantity + QUANTITY_TOLERANCE < order.quantity:
            if self.records_observed():
                # The partial record describes the filled slice rather than the
                # whole order, so it needs its own copy — built only when
                # something will actually read it.
                self.record(
                    replace(order, quantity=fill_quantity),
                    OrderStatus.PARTIALLY_FILLED,
                    "volume limited",
                )
            else:
                self.order_record_count += 1
            return replace(order, quantity=order.quantity - fill_quantity)
        self._note(order, OrderStatus.FILLED)
        return None

    def activate_children(self, parent_order_id: str) -> list[Order]:
        children = self._held_orders_by_parent.pop(parent_order_id, [])
        for child in children:
            self.add_open_order(child)
        return children

    def cancel_oco_siblings(self, filled_order: Order, *, reason: str = "oco sibling filled") -> list[OrderRecord]:
        if filled_order.oco_group is None:
            return []
        records = []
        # An OCO group's members are usually few (a bracket is two), so this
        # index lookup replaces a scan of every open order in the book,
        # regardless of symbol — cancel_oco_siblings runs once per fill, so
        # that scan cost was paid every fill for the life of the run.
        for order in self.open_orders_for_oco_group(filled_order.oco_group):
            if order.id != filled_order.id:
                self.remove_open_order(order.id or "")
                self._cancel_children(order.id or "", reason=reason)
                records.append(self.record(order, OrderStatus.CANCELLED, reason))
        for order in self._held_orders():
            if order.id != filled_order.id and order.oco_group == filled_order.oco_group:
                self._remove_held_order(order.id or "")
                records.append(self.record(order, OrderStatus.CANCELLED, reason))
        return records

    def update_trailing_stop(self, order: Order, bar: Bar) -> Order:
        if order.order_type != OrderType.TRAILING_STOP:
            return order
        if order.trail_amount is not None:
            candidate = bar.high - order.trail_amount if order.side == Side.SELL else bar.low + order.trail_amount
        elif order.trail_percent is not None:
            candidate = bar.high * (1.0 - order.trail_percent) if order.side == Side.SELL else bar.low * (1.0 + order.trail_percent)
        else:
            return order
        stop_price = order.stop_price
        if stop_price is None:
            stop_price = candidate
        elif order.side == Side.SELL:
            stop_price = max(stop_price, candidate)
        else:
            stop_price = min(stop_price, candidate)
        if stop_price == order.stop_price:
            return order
        updated = replace(order, stop_price=stop_price)
        if order.id is not None:
            self.update_open_order(updated)
        return updated

    def expire(self, order: Order, *, reason: str = "order expired") -> OrderRecord:
        if order.id is not None:
            self.remove_open_order(order.id)
            self._cancel_children(order.id, reason=reason)
        return self.record(order, OrderStatus.EXPIRED, reason)

    def expire_orders(self, timestamp: datetime, *, reason: str = "order expired") -> list[OrderRecord]:
        if self._expirable_open_order_count == 0:
            return []
        records = []
        candidates = [self._open_orders_by_id[key] for key in self._expirable_order_ids]
        for order in candidates:
            if self.is_expired(order, timestamp):
                records.append(self.expire(order, reason=reason))
        return records

    @staticmethod
    def validate_order(order: Order) -> str | None:
        if order.quantity <= 0:
            return "order quantity must be positive"
        if order.order_type == OrderType.LIMIT and order.limit_price is None:
            return "limit orders require limit_price"
        if order.order_type == OrderType.STOP and order.stop_price is None:
            return "stop orders require stop_price"
        if order.order_type == OrderType.STOP_LIMIT:
            if order.stop_price is None:
                return "stop-limit orders require stop_price"
            if order.limit_price is None:
                return "stop-limit orders require limit_price"
        if order.order_type == OrderType.TRAILING_STOP:
            if order.trail_amount is None and order.trail_percent is None:
                return "trailing stop orders require trail_amount or trail_percent"
            if order.trail_amount is not None and order.trail_percent is not None:
                return "trailing stop orders require only one of trail_amount or trail_percent"
            if order.trail_amount is not None and order.trail_amount <= 0:
                return "trail_amount must be positive"
            if order.trail_percent is not None and not 0 < order.trail_percent < 1:
                return "trail_percent must be in (0, 1)"
        if order.limit_price is not None and order.limit_price <= 0:
            return "limit_price must be positive"
        if order.stop_price is not None and order.stop_price <= 0:
            return "stop_price must be positive"
        if order.expires_at is not None and order.expires_at < order.timestamp:
            return "expires_at must be greater than or equal to order timestamp"
        if order.time_in_force == TimeInForce.GTD and order.expires_at is None:
            return "GTD orders require expires_at"
        if order.time_in_force == TimeInForce.DAY and order.expires_at is not None:
            return "DAY orders must not set expires_at"
        if order.parent_id is not None and order.parent_id == order.id:
            return "parent_id must not equal order id"
        return None

    @staticmethod
    def is_expired(order: Order, timestamp: datetime) -> bool:
        if order.time_in_force == TimeInForce.DAY:
            return timestamp.date() > order.timestamp.date()
        return order.expires_at is not None and timestamp > order.expires_at

    def _cancel_children(self, parent_order_id: str, *, reason: str) -> list[OrderRecord]:
        children = self._held_orders_by_parent.pop(parent_order_id, [])
        return [self.record(child, OrderStatus.CANCELLED, reason) for child in children]

    def _held_orders(self) -> list[Order]:
        return [order for orders in self._held_orders_by_parent.values() for order in orders]

    def _held_orders_for_symbol(self, symbol: str) -> list[Order]:
        symbol = symbol.upper()
        return [order for order in self._held_orders() if order.symbol.upper() == symbol]

    def _held_order(self, order_id: str) -> Order | None:
        for orders in self._held_orders_by_parent.values():
            for order in orders:
                if order.id == order_id:
                    return order
        return None

    def _replace_held_order(self, order_id: str, replacement: Order) -> bool:
        for parent_id, orders in list(self._held_orders_by_parent.items()):
            remaining = []
            replaced = False
            for order in orders:
                if order.id == order_id:
                    replaced = True
                    continue
                remaining.append(order)
            if not replaced:
                continue
            if remaining:
                self._held_orders_by_parent[parent_id] = remaining
            else:
                self._held_orders_by_parent.pop(parent_id, None)
            if replacement.parent_id is not None and replacement.parent_id in self._open_orders_by_id:
                self._held_orders_by_parent[replacement.parent_id].append(replacement)
            else:
                self.add_open_order(replacement)
            return True
        return False

    def _remove_held_order(self, order_id: str) -> Order | None:
        for parent_id, orders in list(self._held_orders_by_parent.items()):
            remaining = []
            removed = None
            for order in orders:
                if order.id == order_id and removed is None:
                    removed = order
                else:
                    remaining.append(order)
            if removed is not None:
                if remaining:
                    self._held_orders_by_parent[parent_id] = remaining
                else:
                    self._held_orders_by_parent.pop(parent_id, None)
                return removed
        return None
