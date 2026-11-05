"""Order execution logic for the event-driven broker."""

from __future__ import annotations

from dataclasses import replace

from ..contracts import Bar, Fill, FillPrice, Order, OrderType, Side
from .costs import BasisPointSlippage, CommissionModel, PercentCommission, SlippageModel
from .ledger import Ledger
from .matching import MatchingModel
from .risk import RiskContext, RiskEngine


class ExecutionHandler:
    """Evaluate order triggers, prices, costs, risk, and ledger fills."""

    def __init__(
        self,
        *,
        ledger: Ledger,
        risk_engine: RiskEngine,
        commission_model: CommissionModel | None = None,
        slippage_model: SlippageModel | None = None,
        commission_bps: float = 0.0,
        slippage_bps: float = 0.0,
        fill_price: FillPrice = FillPrice.OPEN,
        allow_short: bool = True,
        matching_model: MatchingModel | None = None,
    ) -> None:
        self.ledger = ledger
        self.risk_engine = risk_engine
        self.commission_model = commission_model or PercentCommission(commission_bps)
        self.slippage_model = slippage_model or BasisPointSlippage(slippage_bps)
        self.fill_price = fill_price
        self.allow_short = allow_short
        self.matching_model = matching_model
        # Cache eligibility for the fixed cost models; mutable risk rules
        # must still be checked before each fast fill.
        self._zero_cost_market_path = (
            fill_price in (FillPrice.CLOSE, FillPrice.OPEN)
            and matching_model is None
            and type(self.commission_model) is PercentCommission
            and self.commission_model.bps == 0
            and self.commission_model.minimum == 0
            and type(self.slippage_model) is BasisPointSlippage
            and self.slippage_model.bps == 0
        )

    def try_fill(
        self,
        order: Order,
        bar: Bar,
        *,
        force: bool = False,
        max_quantity: float | None = None,
    ) -> Fill | None:
        if not force and self._can_fast_fill_market_order(order, max_quantity):
            return self._fast_fill_market_order(order, bar)
        base_price = self.base_fill_price(bar)
        if not force and self.matching_model is not None:
            matched = self.matching_model.match(
                order,
                bar,
                default_price=base_price,
                max_quantity=max_quantity,
            )
            if matched is None:
                return None
            matched_price, matched_quantity = matched
            return self.apply_fill(order, bar, matched_price, quantity=matched_quantity)
        if force:
            return self.apply_fill(order, bar, base_price, quantity=max_quantity)
        if order.order_type == OrderType.LIMIT:
            base_price = self.limit_fill_price(order, bar, base_price)
            if base_price is None:
                return None
        elif order.order_type in {OrderType.STOP, OrderType.TRAILING_STOP}:
            base_price = self.stop_fill_price(order, bar, base_price)
            if base_price is None:
                return None
        elif order.order_type == OrderType.STOP_LIMIT:
            base_price = self.stop_limit_fill_price(order, bar, base_price)
            if base_price is None:
                return None
        return self.apply_fill(order, bar, base_price, quantity=max_quantity)

    def apply_fill(
        self,
        order: Order,
        bar: Bar,
        fill_price: float,
        *,
        quantity: float | None = None,
    ) -> Fill:
        fill_quantity = order.quantity if quantity is None else min(quantity, order.quantity)
        if fill_quantity <= 0:
            raise ValueError("fill quantity must be positive")
        fill_order = order if fill_quantity == order.quantity else replace(order, quantity=fill_quantity)
        slipped_price = self.slippage_model.adjust_price(fill_order, bar, fill_price)
        multiplier = self.ledger.contract_multiplier(order.symbol)
        notional = self.ledger.notional(order.symbol, fill_quantity, fill_price)
        slipped_notional = self.ledger.notional(order.symbol, fill_quantity, slipped_price)
        commission = self.commission_model.calculate(
            fill_order,
            slipped_price,
            contract_multiplier=multiplier,
        )

        reason = self.ledger.can_fill(
            side=order.side,
            symbol=order.symbol,
            quantity=fill_quantity,
            value=slipped_notional,
            commission=commission,
            allow_short=self.allow_short,
        )
        if reason is not None:
            raise ValueError(reason)

        reason = self.risk_engine.check(
            RiskContext(order=fill_order, state=self.ledger.state, price=slipped_price)
        )
        if reason is not None:
            raise ValueError(reason)

        fill = Fill(
            symbol=order.symbol,
            timestamp=bar.timestamp,
            side=order.side,
            quantity=fill_quantity,
            price=slipped_price,
            commission=commission,
            slippage=abs(slipped_notional - notional),
            order_id=order.id,
        )
        self.ledger.apply_fill(fill)
        return fill

    def _can_fast_fill_market_order(self, order: Order, max_quantity: float | None) -> bool:
        return (
            self._zero_cost_market_path
            and type(self.risk_engine) is RiskEngine
            and not self.risk_engine.has_rules
            and order.order_type == OrderType.MARKET
            and (max_quantity is None or order.quantity <= max_quantity)
        )

    def _fast_fill_market_order(self, order: Order, bar: Bar) -> Fill:
        price = bar.open if self.fill_price is FillPrice.OPEN else bar.return_close
        reason = self.ledger.can_fill(
            side=order.side,
            symbol=order.symbol,
            quantity=order.quantity,
            value=self.ledger.notional(order.symbol, order.quantity, price),
            commission=0.0,
            allow_short=self.allow_short,
        )
        if reason is not None:
            raise ValueError(reason)
        fill = Fill(
            symbol=order.symbol,
            timestamp=bar.timestamp,
            side=order.side,
            quantity=order.quantity,
            price=price,
            order_id=order.id,
        )
        self.ledger.apply_fill(fill)
        return fill

    def limit_fill_price(self, order: Order, bar: Bar, base_price: float) -> float | None:
        if order.limit_price is None:
            return None
        if order.side == Side.BUY and bar.low > order.limit_price:
            return None
        if order.side == Side.SELL and bar.high < order.limit_price:
            return None
        return min(base_price, order.limit_price) if order.side == Side.BUY else max(
            base_price,
            order.limit_price,
        )

    def stop_fill_price(self, order: Order, bar: Bar, base_price: float) -> float | None:
        if order.stop_price is None or not self.stop_triggered(order, bar):
            return None
        if order.side == Side.BUY:
            return max(base_price, order.stop_price)
        return min(base_price, order.stop_price)

    def stop_limit_fill_price(self, order: Order, bar: Bar, base_price: float) -> float | None:
        if order.stop_price is None or order.limit_price is None:
            return None
        if not self.stop_triggered(order, bar):
            return None
        stop_price = self.stop_fill_price(order, bar, base_price)
        if stop_price is None:
            return None
        return self.limit_fill_price(order, bar, stop_price)

    @staticmethod
    def stop_triggered(order: Order, bar: Bar) -> bool:
        if order.stop_price is None:
            return False
        if order.side == Side.BUY:
            return bar.high >= order.stop_price
        return bar.low <= order.stop_price

    def base_fill_price(self, bar: Bar) -> float:
        if self.fill_price == FillPrice.OPEN:
            return bar.open
        if self.fill_price == FillPrice.CLOSE:
            return bar.return_close
        if self.fill_price == FillPrice.MID:
            return (bar.high + bar.low) / 2.0
        if self.fill_price == FillPrice.HLC3:
            return (bar.high + bar.low + bar.return_close) / 3.0
        if self.fill_price == FillPrice.OHLC4:
            return (bar.open + bar.high + bar.low + bar.return_close) / 4.0
        if self.fill_price == FillPrice.VWAP:
            if bar.vwap is None:
                raise ValueError("VWAP fill price requires bar.vwap")
            return bar.vwap
        raise ValueError(f"Unsupported fill price model: {self.fill_price}")
