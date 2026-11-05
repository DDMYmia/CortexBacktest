"""Backtrader-style position sizing helpers."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from ..contracts import Bar
from .broker import BrokerSimulator


class Sizer:
    def size(self, *, broker: BrokerSimulator, bar: Bar) -> float:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class FixedSizeSizer(Sizer):
    quantity: float

    def size(self, *, broker: BrokerSimulator, bar: Bar) -> float:
        if not isfinite(self.quantity) or self.quantity <= 0:
            raise ValueError("quantity must be finite and positive")
        return self.quantity


@dataclass(frozen=True, slots=True)
class PercentCashSizer(Sizer):
    percent: float

    def size(self, *, broker: BrokerSimulator, bar: Bar) -> float:
        _validate_percent(self.percent)
        return (broker.state.cash * self.percent) / bar.return_close


@dataclass(frozen=True, slots=True)
class TargetPercentSizer(Sizer):
    target_percent: float

    def size(self, *, broker: BrokerSimulator, bar: Bar) -> float:
        _validate_percent(abs(self.target_percent))
        equity = broker.state.equity({bar.symbol: bar.return_close})
        target_value = equity * self.target_percent
        current = broker.state.positions.get(bar.symbol.upper())
        current_value = 0.0 if current is None else current.quantity * bar.return_close
        return (target_value - current_value) / bar.return_close


def _validate_percent(percent: float) -> None:
    if not 0 <= percent <= 1:
        raise ValueError("percent must be in [0, 1]")
