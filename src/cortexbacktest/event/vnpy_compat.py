"""vn.py-style CTA compatibility helpers for offline backtesting."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import groupby
from typing import Callable

from ..contracts import Bar, Contract, Tick
from .strategy import Strategy


class CtaTemplate(Strategy):
    """Small vn.py-style CTA template mapped onto the CortexBacktest Strategy API."""

    author = ""

    def on_init(self) -> None:
        pass

    def on_start(self) -> None:
        pass

    def on_stop(self) -> None:
        pass

    def initialize(self) -> None:
        self.on_init()
        self.on_start()

    def finalize(self) -> None:
        self.on_stop()


class PortfolioStrategyTemplate(Strategy):
    """vn.py-style portfolio strategy template for multi-symbol bar slices."""

    author = ""

    def on_init(self) -> None:
        pass

    def on_start(self) -> None:
        pass

    def on_stop(self) -> None:
        pass

    def on_portfolio_bars(self, bars: dict[str, Bar]):
        return []

    def initialize(self) -> None:
        self.on_init()
        self.on_start()

    def on_bars(self, bars: list[Bar]):
        returned = self.on_portfolio_bars({bar.symbol.upper(): bar for bar in bars})
        orders = list(returned) if returned else []
        orders.extend(self._drain_pending_orders())
        return orders

    def finalize(self) -> None:
        self.on_stop()


@dataclass(frozen=True, slots=True)
class VnpyBacktestingConfig:
    contract: Contract
    interval: str = "1m"
    capital: float = 100_000.0
    rate: float = 0.0
    slippage: float = 0.0
    size: float = 1.0
    price_tick: float = 0.0


class BarGenerator:
    """Aggregate ticks into minute bars and optional window bars."""

    def __init__(
        self,
        on_bar: Callable[[Bar], None],
        *,
        window: int = 1,
        on_window_bar: Callable[[Bar], None] | None = None,
    ) -> None:
        if window < 1:
            raise ValueError("window must be >= 1")
        self.on_bar = on_bar
        self.window = window
        self.on_window_bar = on_window_bar
        self._bar: Bar | None = None
        self._window_bars: list[Bar] = []

    def update_tick(self, tick: Tick) -> None:
        minute = tick.timestamp.replace(second=0, microsecond=0)
        if self._bar is not None and self._bar.timestamp != minute:
            self._emit_bar()
        price = tick.last_price
        volume = tick.volume or 0.0
        if self._bar is None:
            self._bar = Bar(tick.symbol.upper(), minute, price, price, price, price, volume=volume)
            return
        self._bar = Bar(
            symbol=self._bar.symbol,
            timestamp=self._bar.timestamp,
            open=self._bar.open,
            high=max(self._bar.high, price),
            low=min(self._bar.low, price),
            close=price,
            volume=(self._bar.volume or 0.0) + volume,
        )

    def close(self) -> None:
        if self._bar is not None:
            self._emit_bar()

    def _emit_bar(self) -> None:
        if self._bar is None:
            return
        bar = self._bar
        self._bar = None
        self.on_bar(bar)
        if self.on_window_bar is None:
            return
        self._window_bars.append(bar)
        if len(self._window_bars) >= self.window:
            self.on_window_bar(_merge_bars(self._window_bars))
            self._window_bars = []


def ticks_to_bars(ticks: list[Tick]) -> list[Bar]:
    bars: list[Bar] = []
    ordered = sorted(ticks, key=lambda item: (item.symbol.upper(), item.timestamp))
    for _, group in groupby(ordered, key=lambda item: item.symbol.upper()):
        generator = BarGenerator(bars.append)
        for tick in group:
            generator.update_tick(tick)
        generator.close()
    return sorted(bars, key=lambda bar: (bar.timestamp, bar.symbol))


def _merge_bars(bars: list[Bar]) -> Bar:
    first = bars[0]
    volume = sum(bar.volume or 0.0 for bar in bars)
    return Bar(
        symbol=first.symbol,
        timestamp=first.timestamp,
        open=first.open,
        high=max(bar.high for bar in bars),
        low=min(bar.low for bar in bars),
        close=bars[-1].close,
        volume=volume,
    )
