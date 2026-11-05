"""Compare an event-driven backtest against a benchmark equity curve."""

from __future__ import annotations

from datetime import UTC, datetime

import polars as pl

from cortexbacktest import BacktestEngine, Bar, Order, Side, Strategy


class BuyAndHoldStrategy(Strategy):
    def __init__(self) -> None:
        self.submitted = False

    def on_bar(self, bar: Bar):
        if self.submitted:
            return []
        self.submitted = True
        return [Order(bar.symbol, bar.timestamp, Side.BUY, 100)]


def main() -> None:
    bars = [
        Bar("AAPL", datetime(2024, 1, 1, tzinfo=UTC), 100, 100, 100, 100),
        Bar("AAPL", datetime(2024, 1, 2, tzinfo=UTC), 104, 104, 104, 104),
        Bar("AAPL", datetime(2024, 1, 3, tzinfo=UTC), 102, 102, 102, 102),
        Bar("AAPL", datetime(2024, 1, 4, tzinfo=UTC), 108, 108, 108, 108),
    ]
    result = BacktestEngine().run(bars, BuyAndHoldStrategy())

    # A benchmark is a timestamp/equity Polars frame, same shape as the
    # engine's own equity curve.
    benchmark = pl.DataFrame(
        {
            "timestamp": [datetime(2024, 1, day, tzinfo=UTC) for day in (1, 2, 3, 4)],
            "equity": [100_000.0, 101_000.0, 100_500.0, 102_000.0],
        }
    )
    report = result.performance.benchmark(benchmark, rolling_window=2)

    print("CortexBacktest - benchmark report demo")
    for name, value in report.summary.items():
        print(f"  {name:24s} {value: .4f}")
    print()
    print(report.aligned_returns)
    print()
    print(report.rolling_beta)


if __name__ == "__main__":
    main()
