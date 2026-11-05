"""Optional execution matching models."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..contracts import Bar, FillPrice, Order, OrderType, Side


class MatchingModel:
    """Resolve trigger, fill price, and max quantity for an order/bar pair."""

    def match(
        self,
        order: Order,
        bar: Bar,
        *,
        default_price: float,
        max_quantity: float | None = None,
    ) -> tuple[float, float | None] | None:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class BarLimitMatching(MatchingModel):
    """Limit/stop-aware OHLC matching with optional queue participation."""

    fill_price: FillPrice = FillPrice.CLOSE
    price_improvement: bool = True
    participation: float = 1.0

    def __post_init__(self) -> None:
        if self.participation <= 0 or self.participation > 1:
            raise ValueError("participation must be in (0, 1]")

    def match(
        self,
        order: Order,
        bar: Bar,
        *,
        default_price: float,
        max_quantity: float | None = None,
    ) -> tuple[float, float | None] | None:
        price = self._trigger_price(order, bar, default_price)
        if price is None:
            return None
        quantity = _available_quantity(bar, self.participation, max_quantity)
        return price, quantity

    def _trigger_price(self, order: Order, bar: Bar, default_price: float) -> float | None:
        if order.order_type == OrderType.MARKET:
            return default_price
        if order.order_type == OrderType.LIMIT:
            if order.limit_price is None:
                return None
            if order.side == Side.BUY and bar.low > order.limit_price:
                return None
            if order.side == Side.SELL and bar.high < order.limit_price:
                return None
            if not self.price_improvement:
                return order.limit_price
            return min(default_price, order.limit_price) if order.side == Side.BUY else max(
                default_price,
                order.limit_price,
            )
        if order.order_type in {OrderType.STOP, OrderType.TRAILING_STOP}:
            if order.stop_price is None:
                return None
            if order.side == Side.BUY and bar.high < order.stop_price:
                return None
            if order.side == Side.SELL and bar.low > order.stop_price:
                return None
            return max(default_price, order.stop_price) if order.side == Side.BUY else min(
                default_price,
                order.stop_price,
            )
        if order.order_type == OrderType.STOP_LIMIT:
            if order.stop_price is None or order.limit_price is None:
                return None
            stop = self._trigger_price(
                Order(
                    symbol=order.symbol,
                    timestamp=order.timestamp,
                    side=order.side,
                    quantity=order.quantity,
                    order_type=OrderType.STOP,
                    stop_price=order.stop_price,
                ),
                bar,
                default_price,
            )
            if stop is None:
                return None
            return self._trigger_price(
                Order(
                    symbol=order.symbol,
                    timestamp=order.timestamp,
                    side=order.side,
                    quantity=order.quantity,
                    order_type=OrderType.LIMIT,
                    limit_price=order.limit_price,
                ),
                bar,
                stop,
            )
        return None


@dataclass(frozen=True, slots=True)
class BidAskSpreadMatching(MatchingModel):
    """Market/limit matching against a synthetic bid/ask spread."""

    spread_bps: float = 0.0
    participation: float = 1.0

    def __post_init__(self) -> None:
        if self.spread_bps < 0:
            raise ValueError("spread_bps must be non-negative")
        if self.participation <= 0 or self.participation > 1:
            raise ValueError("participation must be in (0, 1]")

    def match(
        self,
        order: Order,
        bar: Bar,
        *,
        default_price: float,
        max_quantity: float | None = None,
    ) -> tuple[float, float | None] | None:
        half_spread = default_price * (self.spread_bps / 10_000.0) / 2.0
        ask = default_price + half_spread
        bid = default_price - half_spread
        price = ask if order.side == Side.BUY else bid
        if order.order_type == OrderType.LIMIT and order.limit_price is not None:
            # The bid/ask is derived from one price of the bar, so a limit that
            # is not marketable against that quote may still have traded during
            # the bar. Fall back to the bar's range, as BarLimitMatching does,
            # and cap the fill at the limit price.
            if order.side == Side.BUY and order.limit_price < ask:
                if bar.low > order.limit_price:
                    return None
                price = order.limit_price
            elif order.side == Side.SELL and order.limit_price > bid:
                if bar.high < order.limit_price:
                    return None
                price = order.limit_price
        elif order.order_type != OrderType.MARKET:
            return BarLimitMatching(participation=self.participation).match(
                order,
                bar,
                default_price=default_price,
                max_quantity=max_quantity,
            )
        return price, _available_quantity(bar, self.participation, max_quantity)


@dataclass(frozen=True, slots=True)
class MarketStateMatching(MatchingModel):
    """Apply halt and daily price-limit rules around another matching model.

    A bar locked at its upper limit cannot fill buys; a bar locked at its lower
    limit cannot fill sells. Orders on halted bars remain open for later bars.
    """

    base: MatchingModel = field(default_factory=BarLimitMatching)
    tolerance: float = 1e-12

    def __post_init__(self) -> None:
        if self.tolerance < 0:
            raise ValueError("tolerance must be non-negative")

    def match(
        self,
        order: Order,
        bar: Bar,
        *,
        default_price: float,
        max_quantity: float | None = None,
    ) -> tuple[float, float | None] | None:
        if bar.halted:
            return None
        if order.side == Side.BUY and self._locked_at(bar, bar.limit_up):
            return None
        if order.side == Side.SELL and self._locked_at(bar, bar.limit_down):
            return None
        matched = self.base.match(
            order,
            bar,
            default_price=default_price,
            max_quantity=max_quantity,
        )
        if matched is None:
            return None
        price, quantity = matched
        if bar.limit_up is not None and price > bar.limit_up + self.tolerance:
            return None
        if bar.limit_down is not None and price < bar.limit_down - self.tolerance:
            return None
        return price, quantity

    def _locked_at(self, bar: Bar, limit: float | None) -> bool:
        if limit is None:
            return False
        return (
            abs(bar.high - limit) <= self.tolerance
            and abs(bar.low - limit) <= self.tolerance
        )


def _available_quantity(bar: Bar, participation: float, max_quantity: float | None) -> float | None:
    if bar.volume is None:
        return max_quantity
    quantity = float(bar.volume) * participation
    return quantity if max_quantity is None else min(max_quantity, quantity)
