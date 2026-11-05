"""Small migration helpers for Backtrader users."""

from __future__ import annotations

from dataclasses import dataclass

from .strategy import Strategy


class BacktraderStrategyMixin(Strategy):
    """Expose familiar Backtrader method aliases on CortexBacktest strategies."""

    def next(self):
        """Override in migrated strategies; called from `on_bars` by default."""
        return []

    def start(self) -> None:
        """Backtrader alias for `initialize`."""

    def stop(self) -> None:
        """Backtrader alias for `finalize`."""

    def initialize(self) -> None:
        self.start()

    def on_bars(self, bars):
        self._set_market_slice(bars)
        returned = self.next()
        orders = list(returned) if returned else []
        orders.extend(self._drain_pending_orders())
        return orders

    def finalize(self) -> None:
        self.stop()


@dataclass(frozen=True, slots=True)
class BacktraderMigrationNote:
    backtrader_name: str
    cortex_name: str
    note: str


MIGRATION_NOTES = [
    BacktraderMigrationNote("Cerebro", "BacktestEngine", "CortexBacktest keeps orchestration small and explicit."),
    BacktraderMigrationNote("Strategy.next", "Strategy.on_bar/on_bars", "Use on_bars for market-slice logic."),
    BacktraderMigrationNote("Sizer", "event.sizers", "Use explicit sizing helpers before queuing orders."),
    BacktraderMigrationNote("Analyzer", "EventAnalyzer", "Analyzers are opt-in and skipped on the hot path when absent."),
    BacktraderMigrationNote("Data resample/replay", "resample_bars/replay_bars", "Use data-layer helpers before running the engine."),
]
