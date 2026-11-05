"""Reusable strategy templates."""

from __future__ import annotations

from collections.abc import Callable

from ..contracts import QUANTITY_TOLERANCE, Bar, Order
from .strategy import Strategy

SignalFunc = Callable[[Bar, Strategy], int | float | None]


class SignalStrategy(Strategy):
    """Minimal signal-to-order strategy template.

    A positive signal buys, a negative signal sells, and zero/None does nothing.
    """

    def __init__(self, signal: SignalFunc, *, quantity: float = 1.0) -> None:
        super().__init__()
        if quantity <= 0:
            raise ValueError("quantity must be positive")
        self.signal = signal
        self.quantity = quantity

    def on_bar(self, bar: Bar) -> list[Order]:
        value = self.signal(bar, self)
        if value is None or abs(value) <= QUANTITY_TOLERANCE:
            return []
        if value > 0:
            self.buy(bar.symbol, quantity=self.quantity)
        else:
            self.sell(bar.symbol, quantity=self.quantity)
        return []


class BuyAndHoldStrategy(Strategy):
    """Buy each configured symbol once on its first bar."""

    def __init__(self, *, quantity: float = 1.0) -> None:
        super().__init__()
        if quantity <= 0:
            raise ValueError("quantity must be positive")
        self.quantity = quantity
        self._bought: set[str] = set()

    def on_bar(self, bar: Bar) -> list[Order]:
        if bar.symbol.upper() in self._bought:
            return []
        self._bought.add(bar.symbol.upper())
        self.buy(bar.symbol, quantity=self.quantity)
        return []
