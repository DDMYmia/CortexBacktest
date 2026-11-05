"""Strategy API for event-driven backtests."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from math import isfinite
from typing import TYPE_CHECKING, Any

from ..contracts import QUANTITY_TOLERANCE, Bar, Order, OrderRecord, OrderStatus, OrderType, Position, Side, TimeInForce

if TYPE_CHECKING:
    from .broker import BrokerSimulator

#: "Leave this field as it is" -- shared with the broker, which reads the same object.
_UNSET = object()


@dataclass(slots=True)
class _AlgoOrderPlan:
    id: str
    symbol: str
    side: Side
    remaining_quantity: float
    mode: str
    total_quantity: float = 0.0
    filled_quantity: float = 0.0
    submitted_quantity: float = 0.0
    cancelled_quantity: float = 0.0
    rejected_quantity: float = 0.0
    outstanding: dict[str, float] = field(default_factory=dict)
    max_retries: int = 3
    retries: int = 0
    stop_reason: str | None = None
    status: str = "active"
    slices_remaining: int | None = None
    display_quantity: float | None = None
    volume_share: float | None = None
    limit_price: float | None = None
    tag: str | None = None
    time_in_force: TimeInForce = TimeInForce.GTC
    last_emit_timestamp: datetime | None = None
    child_order_ids: list[str] = field(default_factory=list)


class Strategy:
    """Base class for event-driven strategies.

    Not an ABC: there is no single hook a strategy must implement. Orders can
    come from `on_bar`, `on_bars`, or `on_before_bars`, and the compatibility
    templates add `on_portfolio_bars` and Backtrader's `next` on top of those.
    Declaring the class abstract while enforcing nothing implied a contract that
    did not exist; `validate_strategy_hooks` performs the check that actually
    matters instead.
    """

    def __init__(self) -> None:
        self._broker: BrokerSimulator | None = None
        self._current_bars: dict[str, Bar] = {}
        self._current_timestamp: datetime | None = None
        self._pending_orders: list[Order] = []
        self._algo_orders: list[_AlgoOrderPlan] = []
        self._algo_history: dict[str, _AlgoOrderPlan] = {}
        self._algo_children: dict[str, _AlgoOrderPlan] = {}
        self._next_client_order_id = 1

    @property
    def broker(self):
        if self._broker is None:
            raise RuntimeError("Strategy broker is only available while a backtest is running")
        return self._broker

    @property
    def bars(self) -> dict[str, Bar]:
        return dict(self._current_bars)

    @property
    def timestamp(self) -> datetime:
        if self._current_timestamp is None:
            raise RuntimeError("Strategy timestamp is only available while a backtest is running")
        return self._current_timestamp

    def initialize(self) -> None:
        """Called once before the backtest loop starts."""

    def before_trading_day(self, trading_day: date, bars: list[Bar]) -> None:
        """Called once before each trading day's first market slice."""

    def notify_order(self, record: OrderRecord) -> None:
        """Called whenever an order status record is created."""

    def on_bar(self, bar: Bar) -> list[Order]:
        """Called for every bar by the default `on_bars` implementation."""
        return []

    def on_bars(self, bars: list[Bar]) -> list[Order]:
        """Called once for each timestamp with all bars in that market slice.

        The queue is drained inline rather than through `_drain_pending_orders`:
        that call replaces the list on every bar whether or not anything was
        queued, and most bars queue nothing. Draining only when there is
        something to drain leaves the semantics identical and allocates nothing
        on the common path.
        """
        orders = self.process_algo_orders(bars)
        for bar in bars:
            returned = self.on_bar(bar)
            if returned:
                orders.extend(returned)
            pending = self._pending_orders
            if pending:
                self._pending_orders = []
                orders.extend(pending)
        pending = self._pending_orders
        if pending:
            self._pending_orders = []
            orders.extend(pending)
        return orders

    def finalize(self) -> None:
        """Called once after the backtest loop ends."""

    def buy(
        self,
        symbol: str | None = None,
        *,
        quantity: float | None = None,
        size: float | None = None,
        limit_price: float | None = None,
        stop_price: float | None = None,
        trail_amount: float | None = None,
        trail_percent: float | None = None,
        tag: str | None = None,
        expires_at: datetime | None = None,
        oco_group: str | None = None,
        parent_id: str | None = None,
        id: str | None = None,
        time_in_force: TimeInForce = TimeInForce.GTC,
    ) -> Order:
        """Queue a buy order for the current market slice."""
        return self._queue_order(
            side=Side.BUY,
            symbol=symbol,
            quantity=_resolve_order_quantity(quantity=quantity, size=size),
            limit_price=limit_price,
            stop_price=stop_price,
            trail_amount=trail_amount,
            trail_percent=trail_percent,
            tag=tag,
            expires_at=expires_at,
            oco_group=oco_group,
            parent_id=parent_id,
            id=id,
            time_in_force=time_in_force,
        )

    def sell(
        self,
        symbol: str | None = None,
        *,
        quantity: float | None = None,
        size: float | None = None,
        limit_price: float | None = None,
        stop_price: float | None = None,
        trail_amount: float | None = None,
        trail_percent: float | None = None,
        tag: str | None = None,
        expires_at: datetime | None = None,
        oco_group: str | None = None,
        parent_id: str | None = None,
        id: str | None = None,
        time_in_force: TimeInForce = TimeInForce.GTC,
    ) -> Order:
        """Queue a sell order for the current market slice."""
        return self._queue_order(
            side=Side.SELL,
            symbol=symbol,
            quantity=_resolve_order_quantity(quantity=quantity, size=size),
            limit_price=limit_price,
            stop_price=stop_price,
            trail_amount=trail_amount,
            trail_percent=trail_percent,
            tag=tag,
            expires_at=expires_at,
            oco_group=oco_group,
            parent_id=parent_id,
            id=id,
            time_in_force=time_in_force,
        )

    def bracket(
        self,
        symbol: str | None = None,
        *,
        quantity: float | None = None,
        size: float | None = None,
        entry_side: Side = Side.BUY,
        entry_limit_price: float | None = None,
        take_profit_price: float | None = None,
        stop_loss_price: float | None = None,
        tag: str | None = "bracket",
    ) -> tuple[Order, Order | None, Order | None]:
        """Queue a parent entry order plus OCO take-profit/stop-loss exits."""
        resolved_symbol = self._resolve_symbol(symbol)
        resolved_quantity = _resolve_order_quantity(quantity=quantity, size=size)
        parent_id = self._client_order_id("B")
        oco_group = self._client_order_id("OCO")
        exit_side = Side.SELL if entry_side == Side.BUY else Side.BUY
        parent = self._queue_order(
            side=entry_side,
            symbol=resolved_symbol,
            quantity=resolved_quantity,
            limit_price=entry_limit_price,
            tag=tag,
            id=parent_id,
        )
        take_profit = None
        if take_profit_price is not None:
            take_profit = self._queue_order(
                side=exit_side,
                symbol=resolved_symbol,
                quantity=resolved_quantity,
                limit_price=take_profit_price,
                tag=tag,
                oco_group=oco_group,
                parent_id=parent_id,
                id=self._client_order_id("B"),
            )
        stop_loss = None
        if stop_loss_price is not None:
            stop_loss = self._queue_order(
                side=exit_side,
                symbol=resolved_symbol,
                quantity=resolved_quantity,
                stop_price=stop_loss_price,
                tag=tag,
                oco_group=oco_group,
                parent_id=parent_id,
                id=self._client_order_id("B"),
            )
        return parent, take_profit, stop_loss

    def twap_order(
        self,
        symbol: str | None = None,
        *,
        total_quantity: float,
        max_retries: int = 3,
        slices: int,
        side: Side = Side.BUY,
        limit_price: float | None = None,
        tag: str | None = "twap",
        time_in_force: TimeInForce = TimeInForce.GTC,
    ) -> str:
        """Start a TWAP-style sliced parent order and emit the first child now."""
        if type(slices) is not int or slices <= 0:
            raise ValueError("slices must be a positive integer")
        return self._add_algo_order(
            symbol=symbol,
            total_quantity=total_quantity,
            max_retries=max_retries,
            mode="twap",
            side=side,
            slices_remaining=slices,
            limit_price=limit_price,
            tag=tag,
            time_in_force=time_in_force,
        )

    def vwap_order(
        self,
        symbol: str | None = None,
        *,
        total_quantity: float,
        max_retries: int = 3,
        volume_share: float,
        side: Side = Side.BUY,
        limit_price: float | None = None,
        tag: str | None = "vwap",
        time_in_force: TimeInForce = TimeInForce.GTC,
    ) -> str:
        """Start a volume-participation VWAP-style parent order."""
        if not 0 < volume_share <= 1:
            raise ValueError("volume_share must be in (0, 1]")
        return self._add_algo_order(
            symbol=symbol,
            total_quantity=total_quantity,
            max_retries=max_retries,
            mode="vwap",
            side=side,
            volume_share=volume_share,
            limit_price=limit_price,
            tag=tag,
            time_in_force=time_in_force,
        )

    def iceberg_order(
        self,
        symbol: str | None = None,
        *,
        total_quantity: float,
        max_retries: int = 3,
        display_quantity: float,
        side: Side = Side.BUY,
        limit_price: float | None = None,
        tag: str | None = "iceberg",
        time_in_force: TimeInForce = TimeInForce.GTC,
    ) -> str:
        """Start an iceberg parent order that exposes one child order at a time."""
        if not isfinite(display_quantity) or display_quantity <= 0:
            raise ValueError("display_quantity must be positive")
        return self._add_algo_order(
            symbol=symbol,
            total_quantity=total_quantity,
            max_retries=max_retries,
            mode="iceberg",
            side=side,
            display_quantity=display_quantity,
            limit_price=limit_price,
            tag=tag,
            time_in_force=time_in_force,
        )

    def process_algo_orders(self, bars: list[Bar] | None = None) -> list[Order]:
        """Emit child orders for active TWAP/VWAP/Iceberg plans."""
        if not self._algo_orders:
            return []
        bar_map = self._current_bars if bars is None else {bar.symbol.upper(): bar for bar in bars}
        orders = []
        active = []
        for plan in self._algo_orders:
            if plan.status != "active":
                continue
            order = self._make_algo_child(plan, bar_map.get(plan.symbol))
            if order is not None:
                orders.append(order)
            if plan.status == "active":
                active.append(plan)
        self._algo_orders = active
        return orders

    def position(self, symbol: str | None = None) -> Position | None:
        """Return the broker position for `symbol`, or the current bar symbol."""
        return self._position(self._resolve_symbol(symbol))

    def close(
        self,
        symbol: str | None = None,
        *,
        quantity: float | None = None,
        size: float | None = None,
        portion: float | None = None,
        tag: str | None = None,
    ) -> Order | None:
        """Queue an order that closes all or part of the current position."""
        resolved_symbol = self._resolve_symbol(symbol)
        position = self._position(resolved_symbol)
        if position is None or abs(position.quantity) <= QUANTITY_TOLERANCE:
            return None
        close_quantity = _resolve_close_quantity(
            position_quantity=abs(position.quantity),
            quantity=quantity,
            size=size,
            portion=portion,
        )
        if close_quantity <= QUANTITY_TOLERANCE:
            return None
        side = Side.SELL if position.quantity > 0 else Side.BUY
        return self._queue_order(
            side=side,
            symbol=resolved_symbol,
            quantity=close_quantity,
            tag=tag,
        )

    def replace_order(
        self,
        order_id: str,
        *,
        quantity: float | None = None,
        limit_price: float | None | object = _UNSET,
        stop_price: float | None | object = _UNSET,
        trail_amount: float | None | object = _UNSET,
        trail_percent: float | None | object = _UNSET,
        tag: str | None | object = _UNSET,
        expires_at: datetime | None | object = _UNSET,
        time_in_force: TimeInForce | None = None,
        reason: str = "strategy replace",
    ) -> OrderRecord | None:
        """Amend an existing open order and keep its order id stable."""
        plan = self._algo_children.get(order_id)
        if plan is not None and quantity is not None:
            available = plan.remaining_quantity + plan.outstanding.get(order_id, 0.0)
            if not isfinite(quantity) or quantity <= 0 or quantity > available + QUANTITY_TOLERANCE:
                raise ValueError("algo_replacement_exceeds_remaining_target")
        return self.broker.replace_order(
            order_id, quantity=quantity, limit_price=limit_price, stop_price=stop_price,
            trail_amount=trail_amount, trail_percent=trail_percent, tag=tag, expires_at=expires_at,
            time_in_force=time_in_force, timestamp=self.timestamp, reason=reason,
        )

    def order_target_size(
        self,
        symbol: str | None = None,
        *,
        target_size: float,
        tag: str | None = "target_size",
    ) -> Order | None:
        resolved_symbol = self._resolve_symbol(symbol)
        current = self._position(resolved_symbol)
        current_size = 0.0 if current is None else current.quantity
        delta = target_size - current_size
        if abs(delta) <= QUANTITY_TOLERANCE:
            return None
        if delta > 0:
            return self.buy(resolved_symbol, quantity=delta, tag=tag)
        return self.sell(resolved_symbol, quantity=abs(delta), tag=tag)

    def order_target_value(
        self,
        symbol: str | None = None,
        *,
        target_value: float,
        price: float | None = None,
        tag: str | None = "target_value",
    ) -> Order | None:
        resolved_symbol = self._resolve_symbol(symbol)
        resolved_price = self._resolve_price(resolved_symbol, price)
        return self.order_target_size(
            resolved_symbol,
            target_size=target_value / resolved_price,
            tag=tag,
        )

    def order_target_percent(
        self,
        symbol: str | None = None,
        *,
        target_percent: float,
        price: float | None = None,
        tag: str | None = "target_percent",
    ) -> Order | None:
        broker = self.broker
        resolved_symbol = self._resolve_symbol(symbol)
        resolved_price = self._resolve_price(resolved_symbol, price)
        equity = broker.state.equity({resolved_symbol: resolved_price})
        return self.order_target_value(
            resolved_symbol,
            target_value=equity * target_percent,
            price=resolved_price,
            tag=tag,
        )

    def _bind_broker(self, broker: BrokerSimulator) -> None:
        self._broker = broker
        if not hasattr(self, "_current_bars"):
            self._current_bars = {}
        if not hasattr(self, "_current_timestamp"):
            self._current_timestamp = None
        if not hasattr(self, "_pending_orders"):
            self._pending_orders = []
        if not hasattr(self, "_algo_orders"):
            self._algo_orders = []
        if not hasattr(self, "_algo_history"):
            self._algo_history = {}
            self._algo_children = {}
        if not hasattr(self, "_next_client_order_id"):
            self._next_client_order_id = 1

    def _set_market_slice(self, bars: list[Bar]) -> None:
        # Reuse the same dict and skip re-uppercasing: the feed already
        # normalizes bar symbols to upper case (single source of truth), so this
        # avoids one dict allocation and one str.upper() per bar in the hot loop.
        current = self._current_bars
        current.clear()
        for bar in bars:
            current[bar.symbol] = bar
        self._current_timestamp = bars[0].timestamp if bars else None

    def _clear_market_slice(self) -> None:
        self._current_bars = {}
        self._current_timestamp = None
        self._pending_orders = []

    def _drain_pending_orders(self) -> list[Order]:
        orders = self._pending_orders
        self._pending_orders = []
        return orders

    def _queue_order(
        self,
        *,
        side: Side,
        symbol: str | None,
        quantity: float,
        **options: Any,
    ) -> Order:
        order = self._build_order(side=side, symbol=symbol, quantity=quantity, **options)
        self._pending_orders.append(order)
        return order

    def _add_algo_order(
        self,
        *,
        symbol: str | None,
        total_quantity: float,
        max_retries: int = 3,
        mode: str,
        side: Side,
        slices_remaining: int | None = None,
        display_quantity: float | None = None,
        volume_share: float | None = None,
        limit_price: float | None = None,
        tag: str | None = None,
        time_in_force: TimeInForce = TimeInForce.GTC,
    ) -> str:
        if not isfinite(total_quantity) or total_quantity <= 0:
            raise ValueError("total_quantity must be positive")
        if type(max_retries) is not int or not 0 <= max_retries <= 100:
            raise ValueError("max_retries must be an integer in [0, 100]")
        plan = _AlgoOrderPlan(
            id=self._client_order_id(mode.upper()),
            symbol=self._resolve_symbol(symbol),
            side=side,
            remaining_quantity=total_quantity,
            total_quantity=total_quantity,
            max_retries=max_retries,
            mode=mode,
            slices_remaining=slices_remaining,
            display_quantity=display_quantity,
            volume_share=volume_share,
            limit_price=limit_price,
            tag=tag,
            time_in_force=time_in_force,
        )
        self._algo_orders.append(plan)
        self._algo_history[plan.id] = plan
        order = self._make_algo_child(plan, self._current_bars.get(plan.symbol))
        if order is not None:
            self._pending_orders.append(order)
        return plan.id

    def _make_algo_child(self, plan: _AlgoOrderPlan, bar: Bar | None) -> Order | None:
        if plan.status != "active" or plan.stop_reason or plan.remaining_quantity <= QUANTITY_TOLERANCE:
            return None
        if self._current_timestamp is not None and plan.last_emit_timestamp == self._current_timestamp:
            return None
        if plan.mode == "iceberg" and plan.outstanding:
            return None
        quantity = self._algo_child_quantity(plan, bar)
        if quantity <= QUANTITY_TOLERANCE:
            return None
        child_id = self._client_order_id(plan.mode.upper())
        order = self._build_order(
            side=plan.side,
            symbol=plan.symbol,
            quantity=quantity,
            limit_price=plan.limit_price,
            tag=plan.tag,
            id=child_id,
            time_in_force=plan.time_in_force,
        )
        plan.outstanding[child_id] = quantity
        self._algo_children[child_id] = plan
        plan.remaining_quantity -= quantity
        plan.last_emit_timestamp = self._current_timestamp
        plan.child_order_ids.append(child_id)
        return order

    def _algo_child_quantity(self, plan: _AlgoOrderPlan, bar: Bar | None) -> float:
        if plan.mode == "twap":
            slices_remaining = max(plan.slices_remaining or 1, 1)
            quantity = plan.remaining_quantity / slices_remaining
            plan.slices_remaining = slices_remaining - 1
            return min(plan.remaining_quantity, quantity)
        if plan.mode == "vwap":
            if bar is None or bar.volume is None or plan.volume_share is None:
                return 0.0
            return min(plan.remaining_quantity, float(bar.volume) * plan.volume_share)
        if plan.mode == "iceberg":
            return min(plan.remaining_quantity, plan.display_quantity or plan.remaining_quantity)
        return 0.0

    def _observe_algo_order(self, record: OrderRecord) -> None:
        """Internal receipt consumer runs before a subclass's public callback."""
        order_id = record.order.id
        plan = self._algo_children.get(order_id or "")
        if plan is None or order_id not in plan.outstanding:
            return
        outstanding = plan.outstanding[order_id]
        status = record.status
        if status == OrderStatus.SUBMITTED:
            plan.submitted_quantity += record.order.quantity
        elif status in {OrderStatus.PARTIALLY_FILLED, OrderStatus.FILLED}:
            quantity = record.order.quantity
            if quantity > outstanding + QUANTITY_TOLERANCE:
                raise ValueError("algo_fill_exceeds_outstanding")
            plan.filled_quantity += quantity
            outstanding = max(0.0, outstanding - quantity)
            plan.outstanding[order_id] = outstanding
            if outstanding <= QUANTITY_TOLERANCE:
                del plan.outstanding[order_id]
        elif status == OrderStatus.REPLACED:
            delta = record.order.quantity - outstanding
            if delta > plan.remaining_quantity + QUANTITY_TOLERANCE:
                raise ValueError("algo_replacement_exceeds_remaining_target")
            plan.remaining_quantity = max(0.0, plan.remaining_quantity - delta)
            plan.outstanding[order_id] = record.order.quantity
        elif status in {OrderStatus.CANCELLED, OrderStatus.EXPIRED, OrderStatus.REJECTED}:
            del plan.outstanding[order_id]
            if status == OrderStatus.REJECTED:
                plan.rejected_quantity += outstanding
            else:
                plan.cancelled_quantity += outstanding
            plan.remaining_quantity += outstanding
            if not plan.stop_reason:
                if plan.retries < plan.max_retries:
                    plan.retries += 1
                else:
                    plan.stop_reason = "retry_limit_exhausted"
        if not plan.outstanding:
            if plan.filled_quantity + QUANTITY_TOLERANCE >= plan.total_quantity:
                plan.status = "completed"
            elif plan.stop_reason:
                plan.status = "cancelled" if plan.stop_reason == "user_cancelled" else "failed"
        if plan.status != "active":
            self._algo_orders = [active for active in self._algo_orders if active.id != plan.id]

    def cancel_algo_order(self, plan_id: str) -> None:
        """Stop further slices and cancel queued/live children without replenishing them."""
        plan = self._algo_history[plan_id]
        if plan.status != "active":
            return
        plan.stop_reason = "user_cancelled"
        pending = self._pending_orders
        self._pending_orders = [order for order in pending if order.id not in plan.outstanding]
        for order in pending:
            if order.id in plan.outstanding:
                self._observe_algo_order(OrderRecord(order, OrderStatus.CANCELLED, "algo plan cancelled"))
        if self._broker is not None:
            for order_id in list(plan.outstanding):
                self._broker.cancel_order(order_id, reason="algo plan cancelled")
        if not plan.outstanding:
            plan.status = "cancelled"
            self._algo_orders = [active for active in self._algo_orders if active.id != plan.id]

    def _finish_algo_orders(self) -> None:
        """End of data cannot imply execution of still-open children."""
        for plan in self._algo_history.values():
            if plan.status == "active":
                plan.status = "incomplete"
                if plan.stop_reason is None:
                    plan.stop_reason = "open_children_at_end_of_data" if plan.outstanding else "unallocated_quantity_at_end_of_data"
        self._algo_orders = []

    def algo_order_progress(self, plan_id: str | None = None) -> list[dict[str, Any]]:
        """Report original target, outstanding reservation and actual fills separately."""
        plans = [self._algo_history[plan_id]] if plan_id is not None else self._algo_history.values()
        return [{"planId": p.id, "mode": p.mode, "symbol": p.symbol, "status": p.status,
                 "plannedQuantity": p.total_quantity, "submittedQuantity": p.submitted_quantity,
                 "quantityBasis": "planned=filled+open+unallocated; submitted/cancelled/rejected count attempts",
                 "openQuantity": sum(p.outstanding.values()), "filledQuantity": p.filled_quantity,
                 "unallocatedQuantity": p.remaining_quantity, "cancelledQuantity": p.cancelled_quantity,
                 "rejectedQuantity": p.rejected_quantity, "retries": p.retries, "maxRetries": p.max_retries,
                 "reason": p.stop_reason, "childOrderIds": list(p.child_order_ids)}
                for p in plans]

    def _build_order(
        self,
        *,
        side: Side,
        symbol: str | None,
        quantity: float,
        limit_price: float | None = None,
        stop_price: float | None = None,
        tag: str | None = None,
        expires_at: datetime | None = None,
        trail_amount: float | None = None,
        trail_percent: float | None = None,
        oco_group: str | None = None,
        parent_id: str | None = None,
        id: str | None = None,
        time_in_force: TimeInForce = TimeInForce.GTC,
    ) -> Order:
        order_type = _order_type(
            limit_price=limit_price,
            stop_price=stop_price,
            trail_amount=trail_amount,
            trail_percent=trail_percent,
        )
        return Order(
            symbol=self._resolve_symbol(symbol),
            timestamp=self.timestamp,
            side=side,
            quantity=quantity,
            order_type=order_type,
            limit_price=limit_price,
            stop_price=stop_price,
            trail_amount=trail_amount,
            trail_percent=trail_percent,
            tag=tag,
            id=id,
            expires_at=expires_at,
            oco_group=oco_group,
            parent_id=parent_id,
            time_in_force=time_in_force,
        )

    def _client_order_id(self, prefix: str) -> str:
        order_id = f"{prefix}{self._next_client_order_id:08d}"
        self._next_client_order_id += 1
        return order_id

    def _resolve_symbol(self, symbol: str | None) -> str:
        if symbol is not None:
            return symbol.upper()
        if len(self._current_bars) == 1:
            return next(iter(self._current_bars))
        raise ValueError("symbol is required when the current market slice has multiple symbols")

    def _position(self, symbol: str) -> Position | None:
        if self._broker is None:
            return None
        normalized = symbol.upper()
        position = self._broker.state.positions.get(normalized)
        if position is None:
            return None
        return Position(position.symbol, position.quantity, position.last_price)

    def _resolve_price(self, symbol: str, price: float | None) -> float:
        if price is None:
            bar = self._current_bars.get(symbol.upper())
            if bar is None:
                raise ValueError(f"price is required when {symbol} is not in the current market slice")
            price = bar.return_close
        if not isfinite(price) or price <= 0:
            raise ValueError("price must be finite and positive")
        return price


def _order_type(
    *,
    limit_price: float | None,
    stop_price: float | None,
    trail_amount: float | None = None,
    trail_percent: float | None = None,
) -> OrderType:
    if trail_amount is not None or trail_percent is not None:
        return OrderType.TRAILING_STOP
    if limit_price is not None and stop_price is not None:
        return OrderType.STOP_LIMIT
    if limit_price is not None:
        return OrderType.LIMIT
    if stop_price is not None:
        return OrderType.STOP
    return OrderType.MARKET


def _resolve_order_quantity(*, quantity: float | None, size: float | None) -> float:
    if quantity is not None and size is not None:
        raise ValueError("use either quantity or size, not both")
    if size is not None:
        return size
    if quantity is not None:
        return quantity
    return 1.0


def _resolve_close_quantity(
    *,
    position_quantity: float,
    quantity: float | None,
    size: float | None,
    portion: float | None,
) -> float:
    requested_quantity = quantity is not None or size is not None
    if requested_quantity and portion is not None:
        raise ValueError("use either quantity/size or portion, not both")
    if portion is not None:
        if not 0 < portion <= 1:
            raise ValueError("portion must be in (0, 1]")
        return position_quantity * portion
    if requested_quantity:
        return min(_resolve_order_quantity(quantity=quantity, size=size), position_quantity)
    return position_quantity


# Every order-producing hook the engine or a compatibility template dispatches
# through. `Strategy` supplies a working default for each, which is what makes
# the check below necessary.
ORDER_HOOKS = ("on_bar", "on_bars", "on_before_bars")


def validate_strategy_hooks(strategy: object) -> None:
    """Reject a strategy that has no way to place an order.

    Because the base class implements every hook, a subclass that misspells one
    — `on_bars` written as `on_bar_s`, or `on_bar` as `onbar` — runs to
    completion and silently never trades. The result looks like a strategy with
    no edge rather than a typo, and nothing in the output distinguishes the two.
    Overriding any one hook is enough; templates that dispatch to their own hook
    (vn.py's `on_portfolio_bars`, Backtrader's `next`) satisfy this by overriding
    `on_bars` themselves.
    """
    cls = type(strategy)
    for name in ORDER_HOOKS:
        hook = getattr(strategy, name, None)
        if callable(hook) and getattr(hook, "__func__", hook) is not getattr(Strategy, name, None):
            return
    raise TypeError(
        f"{cls.__name__} overrides none of {', '.join(ORDER_HOOKS)}, so it can never place an "
        "order. Implement one of them, or check the spelling of the hook you meant to define."
    )
