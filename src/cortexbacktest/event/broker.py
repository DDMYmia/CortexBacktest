"""Simple backtesting broker."""

from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime
from math import isfinite
from typing import Iterable, Literal, NamedTuple, cast

from ..contracts import (
    QUANTITY_TOLERANCE,
    Bar,
    CorporateAction,
    CorporateActionEvent,
    CorporateActionType,
    Fill,
    FillPrice,
    ForcedLiquidation,
    MarginCall,
    Order,
    OrderRecord,
    OrderType,
    Side,
    TimeInForce,
)
from .corporate_actions import CorporateActionSchedule
from .costs import BasisPointSlippage, CommissionModel, CommissionScheme, PercentCommission, SlippageModel
from .execution import ExecutionHandler
from .financing import CashInterestModel, FinancingCharge, FinancingState, SettlementModel, ShortBorrowFeeModel
from .ledger import AccountState, Ledger
from .matching import MatchingModel
from .orders import OrderManager
from .record_sink import EventRecordSink
from .risk import MarginRequirementRule, RiskContext, RiskEngine, RiskRule
from .strategy import _UNSET, _order_type

BrokerState = AccountState


def _combine_at_date(target_date: date, timestamp: datetime) -> datetime:
    """Re-anchor ``timestamp``'s wall-clock time onto ``target_date``.

    ``datetime.combine(target_date, timestamp.timetz())`` is correct for a
    naive timestamp, UTC, or a `zoneinfo.ZoneInfo` (which recomputes its
    offset per-datetime) — but a pytz timezone caches the offset it was
    localized with, so reusing it verbatim across a DST transition silently
    shifts the result by an hour. pytz zones expose `.localize()`
    specifically to avoid this; use it when available.
    """
    naive = datetime.combine(target_date, timestamp.time())
    tzinfo = timestamp.tzinfo
    if tzinfo is None:
        return naive
    localize = getattr(tzinfo, "localize", None)
    if callable(localize):
        return cast(datetime, localize(naive))
    return naive.replace(tzinfo=tzinfo)


class _AppliedAction(NamedTuple):
    """What one corporate action did to the account, before it becomes an event."""

    quantity_before: float
    quantity_after: float
    cash_amount: float = 0.0
    new_symbol: str | None = None
    tax_amount: float = 0.0
    settles_on: date | None = None


class BrokerSimulator:
    """Bar-based broker simulator for market, limit, stop, and stop-limit orders."""

    def __init__(
        self,
        *,
        initial_cash: float = 100_000.0,
        commission_bps: float = 0.0,
        slippage_bps: float = 0.0,
        commission_model: CommissionModel | None = None,
        commission_scheme: CommissionScheme | None = None,
        slippage_model: SlippageModel | None = None,
        matching_model: MatchingModel | None = None,
        settlement_model: SettlementModel | None = None,
        cash_interest_model: CashInterestModel | None = None,
        short_borrow_fee_model: ShortBorrowFeeModel | None = None,
        contract_multipliers: dict[str, float] | None = None,
        futures_symbols: Iterable[str] | None = None,
        maintenance_margin: float | None = None,
        dividend_withholding_tax: float = 0.0,
        corporate_actions: Iterable[CorporateAction] | CorporateActionSchedule | None = None,
        fill_price: FillPrice = FillPrice.OPEN,
        allow_short: bool = True,
        max_volume_share: float = 1.0,
        risk_rules: list[RiskRule] | None = None,
        intrabar_policy: Literal["submission", "stop_first", "limit_first"] = "submission",
        stop_limit_same_bar: bool = True,
        retain_records: bool = True,
        record_sink: EventRecordSink | None = None,
    ) -> None:
        if intrabar_policy not in {"submission", "stop_first", "limit_first"}:
            raise ValueError("intrabar_policy must be submission, stop_first, or limit_first")
        self.intrabar_policy = intrabar_policy
        self.stop_limit_same_bar = stop_limit_same_bar
        if initial_cash <= 0:
            raise ValueError("initial_cash must be positive")
        if commission_bps < 0 or slippage_bps < 0:
            raise ValueError("cost inputs must be non-negative")
        if not 0 < max_volume_share <= 1:
            raise ValueError("max_volume_share must be in (0, 1]")
        if commission_scheme is not None:
            if commission_model is not None:
                raise ValueError("use either commission_model or commission_scheme, not both")
            commission_model = commission_scheme.commission_model
        rules = list(risk_rules or [])
        if commission_scheme is not None:
            rules.append(MarginRequirementRule(commission_scheme.initial_margin))
        if maintenance_margin is None and commission_scheme is not None:
            maintenance_margin = commission_scheme.maintenance_margin
        if maintenance_margin is not None and (
            not isfinite(maintenance_margin) or maintenance_margin <= 0
        ):
            raise ValueError("maintenance_margin must be positive and finite")
        if not 0.0 <= dividend_withholding_tax < 1.0:
            raise ValueError("dividend_withholding_tax must be in [0, 1)")
        self.ledger = Ledger(
            initial_cash=initial_cash,
            retain_fills=retain_records,
            fill_sink=record_sink.write_fill if record_sink is not None else None,
            settlement_model=settlement_model,
            contract_multipliers=contract_multipliers,
            futures_symbols=futures_symbols,
        )
        self.variation_margin_charges: list[FinancingCharge] | None = []
        self.maintenance_margin = maintenance_margin
        # Default rate for received dividends; a CorporateAction may override it
        # per issuer, since the rate follows the payer's jurisdiction.
        self.dividend_withholding_tax = dividend_withholding_tax
        self.margin_calls: list[MarginCall] | None = []
        self.allow_short = allow_short
        self.max_volume_share = max_volume_share
        self.risk_engine = RiskEngine(rules)
        self.order_manager = OrderManager(
            retain_records=retain_records,
            record_sink=record_sink.write_order_record if record_sink is not None else None,
        )
        self.execution_handler = ExecutionHandler(
            ledger=self.ledger,
            risk_engine=self.risk_engine,
            commission_model=commission_model or PercentCommission(commission_bps),
            slippage_model=slippage_model or BasisPointSlippage(slippage_bps),
            fill_price=fill_price,
            allow_short=allow_short,
            matching_model=matching_model,
        )
        self.cash_interest_model = cash_interest_model
        self.short_borrow_fee_model = short_borrow_fee_model
        self.financing_state = FinancingState(charges=[])
        self.corporate_actions = (
            corporate_actions
            if isinstance(corporate_actions, CorporateActionSchedule)
            else CorporateActionSchedule(corporate_actions or ())
        )
        self.corporate_action_events: list[CorporateActionEvent] | None = []

    @property
    def commission_model(self) -> CommissionModel:
        return self.execution_handler.commission_model

    @property
    def slippage_model(self) -> SlippageModel:
        return self.execution_handler.slippage_model

    @property
    def fill_price(self) -> FillPrice:
        return self.execution_handler.fill_price

    @property
    def open_orders(self) -> list[Order]:
        return self.order_manager.open_orders

    @open_orders.setter
    def open_orders(self, orders: list[Order]) -> None:
        self.order_manager.open_orders = orders

    @property
    def order_records(self) -> list[OrderRecord]:
        return self.order_manager.order_records

    @property
    def state(self) -> BrokerState:
        return self.ledger.state

    @property
    def fills(self) -> list[Fill]:
        return self.ledger.fills

    @property
    def fill_count(self) -> int:
        return self.ledger.fill_count

    @property
    def order_record_count(self) -> int:
        return self.order_manager.order_record_count

    @property
    def submitted_order_count(self) -> int:
        return self.order_manager.submitted_order_count

    def set_record_retention(self, retain_records: bool) -> None:
        self.ledger.retain_fills = retain_records
        self.order_manager.retain_records = retain_records

    def set_record_sink(self, record_sink: EventRecordSink | None) -> None:
        self.ledger.fill_sink = record_sink.write_fill if record_sink is not None else None
        self.order_manager.record_sink = record_sink.write_order_record if record_sink is not None else None

    def drain_order_records(self) -> list[OrderRecord]:
        return self.order_manager.drain_pending_records()

    def submit_order(self, order: Order) -> OrderRecord:
        order = self.order_manager.prepare_order(order)
        reason = self.order_manager.validate_order(order)
        if reason is None:
            reason = self.risk_engine.check(RiskContext(order=order, state=self.state))
        if reason is not None:
            return self.order_manager.reject(order, reason)

        return self.order_manager.submit(order)

    def cancel_order(self, order_id: str, *, reason: str = "cancelled") -> OrderRecord | None:
        """Cancel an open order by id."""
        return self.order_manager.cancel_order(order_id, reason=reason)

    def cancel_symbol(self, symbol: str, *, reason: str = "cancelled") -> list[OrderRecord]:
        """Cancel all open orders for a symbol."""
        return self.order_manager.cancel_symbol(symbol, reason=reason)

    def replace_order(
        self,
        order_id: str,
        *,
        quantity: float | None = None,
        limit_price: float | None | object = _UNSET,
        stop_price: float | None | object = _UNSET,
        trail_amount: float | None | object = _UNSET,
        trail_percent: float | None | object = _UNSET,
        order_type: OrderType | None = None,
        tag: str | None | object = _UNSET,
        expires_at: object = _UNSET,
        time_in_force: TimeInForce | None = None,
        oco_group: str | None | object = _UNSET,
        parent_id: str | None | object = _UNSET,
        timestamp: object = _UNSET,
        reason: str = "replaced",
    ) -> OrderRecord | None:
        """Amend an open or held order without changing its id or submitted count."""
        existing = self.order_manager.get_order(order_id)
        if existing is None:
            return None
        replacement = _replace_order_fields(
            existing,
            quantity=quantity,
            limit_price=limit_price,
            stop_price=stop_price,
            trail_amount=trail_amount,
            trail_percent=trail_percent,
            order_type=order_type,
            tag=tag,
            expires_at=expires_at,
            time_in_force=time_in_force,
            oco_group=oco_group,
            parent_id=parent_id,
            timestamp=timestamp,
        )
        validation_reason = self.order_manager.validate_order(replacement)
        if validation_reason is None:
            validation_reason = self.risk_engine.check(RiskContext(order=replacement, state=self.state))
        if validation_reason is not None:
            return self.order_manager.reject(replacement, validation_reason)
        return self.order_manager.replace_order(order_id, replacement, reason=reason)

    def process_bar(self, bar: Bar, *, expire_orders: bool = True, mark: bool = True) -> list[Fill]:
        if mark:
            self.mark(bar)
        if expire_orders:
            self.expire_orders(bar.timestamp)
        open_order_count = self.order_manager.open_order_count
        if open_order_count == 0:
            return []
        if open_order_count <= 32:
            # Small global books can still span a thousand-symbol universe.
            # Avoid copying/scanning foreign-symbol orders on every other bar.
            orders = self.order_manager.open_orders_for_symbol(bar.symbol)
            if not orders:
                return []
            return self._process_bar_linear(bar, orders)
        return self._process_bar_indexed(bar)

    def _process_bar_linear(
        self, bar: Bar, open_orders: list[Order], *, indexed: bool = False,
    ) -> list[Fill]:
        fills = []
        cancelled_ids = set()
        available_quantity = self._available_fill_quantity(bar)
        # Both book-selection paths share execution and keep their own cancellation checks.
        prefilter = self.execution_handler.matching_model is None
        bar_low = bar.low
        bar_high = bar.high
        for order in self._intrabar_orders(open_orders):
            if not indexed:
                if order.id in cancelled_ids or order.symbol != bar.symbol:
                    continue
            matching_order = self._matching_order(order, bar)
            if matching_order is None or available_quantity <= 0:
                continue
            if prefilter:
                order_type = matching_order.order_type
                if order_type is OrderType.LIMIT:
                    limit_price = order.limit_price
                    if limit_price is not None and (
                        bar_low > limit_price if order.side is Side.BUY else bar_high < limit_price
                    ):
                        continue
                elif order_type is OrderType.STOP or order_type is OrderType.STOP_LIMIT:
                    stop_price = order.stop_price
                    if stop_price is not None and (
                        bar_high < stop_price if order.side is Side.BUY else bar_low > stop_price
                    ):
                        continue
            # The indexed snapshot may contain an OCO sibling cancelled by an earlier fill.
            if indexed and not self.order_manager.is_open(order.id):
                continue
            try:
                fill = self.execution_handler.try_fill(
                    matching_order,
                    bar,
                    max_quantity=available_quantity,
                )
            except ValueError as exc:
                # Keep each path's record/sink ordering around removal on rejection.
                if not indexed:
                    self.order_manager.reject(order, str(exc))
                if order.id is not None:
                    self.order_manager.remove_open_order(order.id)
                if indexed:
                    self.order_manager.reject(order, str(exc))
                continue
            if fill is None:
                # A trailing stop ratchets only after this bar's trigger test:
                # the level protecting bar N is the one set by bar N-1's
                # extreme, so a bar can never trigger a stop that its own
                # high/low just created (look-ahead).
                self.order_manager.update_trailing_stop(order, bar)
                continue
            fills.append(fill)
            available_quantity -= fill.quantity
            remainder = self.order_manager.close_order_after_fill(order, fill.quantity)
            if remainder is not None:
                self.order_manager.update_open_order(remainder)
            else:
                if order.id is not None:
                    self.order_manager.remove_open_order(order.id)
                    # Children join the back of the queue at activation time,
                    # matching the indexed path's time-priority ordering.
                    self.order_manager.activate_children(order.id)
            cancelled = self.order_manager.cancel_oco_siblings(order)
            if not indexed and cancelled:
                # Do not allocate a generator for the common fill without OCO siblings.
                cancelled_ids.update(record.order.id for record in cancelled if record.order.id is not None)
        return fills

    def _matching_order(self, order: Order, bar: Bar) -> Order | None:
        if order.order_type is not OrderType.STOP_LIMIT:
            return order
        if self.order_manager.effective_order_type(order) is OrderType.LIMIT:
            return replace(order, order_type=OrderType.LIMIT)
        if not bar.halted and self.execution_handler.stop_triggered(order, bar):
            self.order_manager.arm_stop_limit(order)
            if not self.stop_limit_same_bar:
                return None
        return order

    def _intrabar_orders(self, orders: list[Order]) -> list[Order]:
        if self.intrabar_policy == "submission":
            return orders
        # Reorder only slots belonging to the same OCO group and symbol;
        # unrelated orders keep their original time priority.
        groups: dict[tuple[str, str], list[int]] = {}
        for index, order in enumerate(orders):
            if order.oco_group is not None:
                groups.setdefault((order.symbol, order.oco_group), []).append(index)
        ordered = list(orders)
        for indices in groups.values():
            def priority(order):
                is_stop = order.order_type in {OrderType.STOP, OrderType.STOP_LIMIT, OrderType.TRAILING_STOP}
                return int(not is_stop) if self.intrabar_policy == "stop_first" else int(is_stop)
            candidates = sorted((orders[index] for index in indices), key=priority)
            for index, order in zip(indices, candidates, strict=True):
                ordered[index] = order
        return ordered

    def before_trading_day(self, timestamp) -> None:
        # Each step below already returns immediately for an account that has
        # none of the feature it implements -- but only after being called, and
        # `accrue_financing` pays a `timestamp.date()` and a date subtraction
        # before it finds out. Five calls per bar to reach five no-ops cost
        # ~15% of a plain long-only equity run (measured: 6.54ms -> 5.57ms over
        # 2,932 bars). The conditions are all cheap attribute reads, so ask them
        # here instead.
        #
        # Read dynamically rather than cached at construction: futures
        # positions and pending settlements appear during a run, and a cached
        # flag would silently stop settling them.
        ledger = self.ledger
        if not (
            ledger.state.futures_symbols
            or ledger.pending_settlements
            or self.corporate_actions
            or self.cash_interest_model is not None
            or self.short_borrow_fee_model is not None
            or self.maintenance_margin is not None
        ):
            return
        # Variation margin settles first, on the prices the previous session
        # closed at: the day's futures P&L is cash before anything this session
        # can spend it, and before margin is checked against it.
        self.settle_variation_margin(timestamp)
        self.apply_corporate_actions(timestamp)
        self.settle_cash(timestamp)
        self.accrue_financing(timestamp)
        # Last: every step above can move the number the requirement is tested
        # against. Yesterday's futures loss has settled into cash, a delisting
        # has closed a holding at its cash-out value, and borrow fees have been
        # charged, so this is the first point where the account is stated the
        # way the broker would see it at the open.
        self.check_maintenance_margin(timestamp)

    def settle_variation_margin(self, timestamp) -> float:
        """Pay the futures mark-to-market into cash and advance the basis."""
        settled = self.ledger.settle_variation_margin()
        if settled and self.variation_margin_charges is not None:
            self.variation_margin_charges.append(
                FinancingCharge(timestamp, settled, "variation_margin")
            )
        return settled

    def check_maintenance_margin(self, timestamp) -> MarginCall | None:
        """Liquidate down to the maintenance requirement, if equity is short of it.

        `MarginRequirementRule` refuses orders an account cannot afford to open.
        This is the other half: a position that was affordable when it opened
        can stop being so without the account trading at all, because the market
        moved. Without this the account simply runs a deficit, and a strategy
        whose real outcome was a liquidation instead reports whatever the
        position did afterwards -- including a recovery it would never have been
        around to collect.

        Returns the call, or `None` when no rate is configured or the account is
        within its requirement.
        """
        rate = self.maintenance_margin
        if rate is None:
            return None
        state = self.state
        if not state.positions:
            return None
        equity = state.equity()
        requirement = state.maintenance_requirement(rate)
        if requirement <= equity:
            return None

        liquidations = self._liquidate_to_requirement(timestamp, rate)
        call = MarginCall(
            timestamp=timestamp,
            equity=equity,
            requirement=requirement,
            liquidations=tuple(liquidations),
            resolved=state.maintenance_requirement(rate) <= state.equity(),
        )
        if self.margin_calls is not None:
            self.margin_calls.append(call)
        return call

    def _liquidate_to_requirement(self, timestamp, rate: float) -> list[ForcedLiquidation]:
        """Close the least that answers the call, largest position first.

        Each position is closed only as far as it takes to clear the shortfall,
        because a broker liquidating more than the deficit would be taking a
        position of its own. Largest first is what makes it "the least": the
        biggest holding frees the most requirement per position touched.

        Everything trades at the mark the breach was measured against. A real
        liquidation happens into the next session at prices nobody has seen yet,
        so this omits exactly the gap risk that makes forced liquidation
        expensive -- the direction of that error is known, and it flatters the
        account.
        """
        state = self.state
        ledger = self.ledger
        liquidations: list[ForcedLiquidation] = []
        ordered = sorted(
            state.positions.values(),
            key=lambda position: abs(
                position.quantity * position.last_price * ledger.contract_multiplier(position.symbol)
            ),
            reverse=True,
        )
        for position in ordered:
            shortfall = state.maintenance_requirement(rate) - state.equity()
            if shortfall <= 0:
                break
            symbol = position.symbol
            price = position.last_price
            multiplier = ledger.contract_multiplier(symbol)
            released_per_unit = abs(price * multiplier) * rate
            if released_per_unit <= 0:
                # A position marked at zero requires no margin and frees none;
                # closing it cannot answer the call.
                continue
            held = position.quantity
            quantity = min(abs(held), shortfall / released_per_unit)
            if quantity < QUANTITY_TOLERANCE:
                continue
            side = Side.SELL if held > 0 else Side.BUY
            commission = self.commission_model.calculate(
                Order(symbol=symbol, timestamp=timestamp, side=side, quantity=quantity),
                price,
                contract_multiplier=multiplier,
            )
            ledger.apply_fill(
                Fill(
                    symbol=symbol,
                    timestamp=timestamp,
                    side=side,
                    quantity=quantity,
                    price=price,
                    commission=commission,
                )
            )
            # The account could not carry what it already held, so orders resting
            # on the symbol -- sized for an account that could -- come off the
            # book with the position.
            self.order_manager.cancel_symbol(symbol, reason="margin call")
            liquidations.append(
                ForcedLiquidation(
                    symbol=symbol,
                    quantity=-quantity if held > 0 else quantity,
                    price=price,
                    commission=commission,
                )
            )
        return liquidations

    def apply_corporate_actions(self, timestamp) -> list[CorporateActionEvent]:
        """Apply this session's corporate actions before any bar is priced."""
        if not self.corporate_actions:
            return []
        events = []
        for action in self.corporate_actions.actions_on(timestamp.date()):
            applied = self._apply_corporate_action(action, timestamp)
            if applied is None:
                # Nothing held in the affected symbol; the action is a no-op.
                continue
            event = CorporateActionEvent(
                timestamp=timestamp,
                symbol=action.symbol,
                action_type=action.action_type,
                quantity_before=applied.quantity_before,
                quantity_after=applied.quantity_after,
                cash_amount=applied.cash_amount,
                new_symbol=applied.new_symbol,
                tax_amount=applied.tax_amount,
                settles_on=applied.settles_on or action.ex_date,
            )
            events.append(event)
            if self.corporate_action_events is not None:
                self.corporate_action_events.append(event)
        return events

    def _apply_corporate_action(
        self, action: CorporateAction, timestamp: datetime
    ) -> _AppliedAction | None:
        """Route one action to the ledger.

        Returns what it did to the account, or `None` when nothing is held in
        the affected symbol.
        """
        action_type = action.action_type
        ledger = self.ledger

        if action_type == CorporateActionType.SPLIT:
            before, after = ledger.apply_split(action.symbol, action.ratio)
            return None if before == 0.0 else _AppliedAction(before, after)

        if action_type == CorporateActionType.STOCK_DIVIDEND:
            # `ratio` extra shares per share held is a split by 1 + ratio, and
            # sharing the split path keeps market value continuous for free.
            before, after = ledger.apply_split(action.symbol, 1.0 + action.ratio)
            return None if before == 0.0 else _AppliedAction(before, after)

        if action_type == CorporateActionType.CASH_DIVIDEND:
            withholding = (
                action.withholding_tax
                if action.withholding_tax is not None
                else self.dividend_withholding_tax
            )
            pay_at = None
            if action.pay_date is not None and action.pay_date > action.ex_date:
                # Anchored to the ex-date session's time of day, so the
                # comparison in `settle_cash` turns on the date rather than on
                # what time the pay-date session happens to open.
                pay_at = _combine_at_date(action.pay_date, timestamp)
            quantity, cash_amount, tax_amount = ledger.apply_cash_dividend(
                action.symbol,
                action.amount,
                withholding_tax=withholding,
                pay_at=pay_at,
            )
            if quantity == 0.0:
                return None
            return _AppliedAction(
                quantity,
                quantity,
                cash_amount,
                tax_amount=tax_amount,
                settles_on=action.pay_date or action.ex_date,
            )

        if action_type == CorporateActionType.DELISTING:
            before, cash_amount = ledger.apply_delisting(action.symbol, action.amount)
            # The position is gone, so any orders resting on it can never fill.
            if before != 0.0:
                self.order_manager.cancel_symbol(action.symbol, reason="delisted")
            return None if before == 0.0 else _AppliedAction(before, 0.0, cash_amount)

        if action_type == CorporateActionType.SYMBOL_CHANGE:
            new_symbol = action.new_symbol or ""
            quantity = ledger.apply_symbol_change(action.symbol, new_symbol)
            # Orders reference the old ticker, which will never trade again.
            if quantity != 0.0:
                self.order_manager.cancel_symbol(action.symbol, reason="symbol changed")
            if quantity == 0.0:
                return None
            return _AppliedAction(quantity, quantity, new_symbol=new_symbol)

        new_symbol = action.new_symbol or ""
        before, received, cash_amount = ledger.apply_merger(
            action.symbol, new_symbol, action.ratio, action.amount
        )
        if before == 0.0:
            return None
        self.order_manager.cancel_symbol(action.symbol, reason="merged")
        return _AppliedAction(before, received, cash_amount, new_symbol=new_symbol)

    def _process_bar_indexed(self, bar: Bar) -> list[Fill]:
        # Resting limit/stop orders whose trigger price is outside this bar's
        # range cannot fill; the checks below mirror limit_fill_price /
        # stop_triggered exactly. Only valid for the default matching path — a
        # custom matching model may fill on conditions of its own.
        prefilter = self.execution_handler.matching_model is None
        bar_low = bar.low
        bar_high = bar.high
        # Same test, asked once for the whole symbol instead of once per
        # resting order: at scale most bars cannot fill anything, and walking
        # the symbol's book to rediscover that is the dominant cost.
        if prefilter and not self.order_manager.symbol_can_trigger(bar.symbol, bar_low, bar_high):
            return []
        return self._process_bar_linear(
            bar, self.order_manager.open_orders_for_symbol(bar.symbol), indexed=True,
        )

    def expire_orders(self, timestamp) -> list[OrderRecord]:
        return self.order_manager.expire_orders(timestamp)

    def settle_cash(self, timestamp) -> float:
        return self.ledger.settle_cash(timestamp)

    def accrue_financing(self, timestamp) -> list[FinancingCharge]:
        current_date = timestamp.date()
        last_date = self.financing_state.last_accrual_date
        self.financing_state.last_accrual_date = current_date
        if last_date is None:
            return []
        days = (current_date - last_date).days
        if days <= 0:
            return []
        charges = []
        if self.cash_interest_model is not None:
            amount = self.cash_interest_model.charge(self.state.cash, days)
            if amount:
                self.ledger.apply_cash_adjustment(amount)
                charges.append(FinancingCharge(timestamp, amount, "cash_interest"))
        if self.short_borrow_fee_model is not None:
            amount = self.short_borrow_fee_model.charge(self.state.positions, days)
            if amount:
                self.ledger.apply_cash_adjustment(amount)
                charges.append(FinancingCharge(timestamp, amount, "short_borrow_fee"))
        if self.financing_state.charges is not None:
            self.financing_state.charges.extend(charges)
        return charges

    def mark(self, bar: Bar) -> None:
        self.ledger.mark(bar)

    def execute(self, order: Order, bar: Bar) -> Fill:
        """Immediately execute an order on `bar`.

        This is kept as a convenience API for tests and simple demos. The full
        event engine uses `submit_order` and `process_bar`.
        """
        if order.symbol != bar.symbol:
            raise ValueError("order symbol must match fill bar symbol")
        reason = self.order_manager.validate_order(order)
        if reason is None:
            reason = self.risk_engine.check(RiskContext(order=order, state=self.state))
        if reason is not None:
            raise ValueError(reason)
        fill = self.execution_handler.try_fill(order, bar, force=True)
        if fill is None:
            raise ValueError("order was not fillable on the supplied bar")
        return fill

    def _available_fill_quantity(self, bar: Bar) -> float:
        if bar.volume is None:
            return float("inf")
        return max(float(bar.volume) * self.max_volume_share, 0.0)

    def equity_snapshot(self, timestamp: datetime) -> dict[str, float | datetime]:
        return self.ledger.equity_snapshot(timestamp)

    def equity_snapshot_row(self, timestamp: datetime) -> tuple:
        """Snapshot values ordered as ledger.EQUITY_SNAPSHOT_COLUMNS."""
        # Overriding this stays supported; the engine notices and keeps the
        # nine-column path for that run rather than deriving anything.
        return self.ledger.equity_snapshot_row(timestamp)


def _replace_order_fields(
    order: Order,
    *,
    quantity: float | None,
    limit_price: float | None | object,
    stop_price: float | None | object,
    trail_amount: float | None | object,
    trail_percent: float | None | object,
    order_type: OrderType | None,
    tag: str | None | object,
    expires_at: object,
    time_in_force: TimeInForce | None,
    oco_group: str | None | object,
    parent_id: str | None | object,
    timestamp: object,
) -> Order:
    updates = {}
    if quantity is not None:
        updates["quantity"] = quantity
    for name, value in (
        ("limit_price", limit_price),
        ("stop_price", stop_price),
        ("trail_amount", trail_amount),
        ("trail_percent", trail_percent),
        ("tag", tag),
        ("expires_at", expires_at),
        ("oco_group", oco_group),
        ("parent_id", parent_id),
        ("timestamp", timestamp),
    ):
        if value is not _UNSET:
            updates[name] = value
    if time_in_force is not None:
        updates["time_in_force"] = time_in_force
    if order_type is None:
        updates["order_type"] = _order_type(
            limit_price=updates.get("limit_price", order.limit_price),
            stop_price=updates.get("stop_price", order.stop_price),
            trail_amount=updates.get("trail_amount", order.trail_amount),
            trail_percent=updates.get("trail_percent", order.trail_percent),
        )
    else:
        updates["order_type"] = order_type
    return replace(order, **updates)


