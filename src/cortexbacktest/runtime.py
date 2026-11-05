"""Application composition of simulation and reporting."""

from typing import cast
from .contracts import Bar
from .event.feed import ColumnarBarFeed
from .event.dispatch import StrategyHooks
from .event.loop import BacktestEngine as EventLoop, ResultRetention, SnapshotFrequency
from .reporting.result import BacktestResult


class BacktestEngine(EventLoop):
    result_type = BacktestResult

    def run(
        self,
        bars: list[Bar] | ColumnarBarFeed,
        strategy: StrategyHooks,
        *,
        assume_sorted: bool = False,
    ) -> BacktestResult:
        return cast(BacktestResult, super().run(bars, strategy, assume_sorted=assume_sorted))


__all__ = ["BacktestEngine", "BacktestResult", "ResultRetention", "SnapshotFrequency"]
