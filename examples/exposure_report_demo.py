"""Inspect positions, exposure, group exposure, and turnover."""

from __future__ import annotations

from datetime import UTC, datetime

from cortexbacktest import BacktestEngine, Bar, Order, Side, Strategy


class RotationStrategy(Strategy):
    def on_bar(self, bar: Bar):
        if bar.symbol == "AAPL" and bar.timestamp.day == 1:
            return [Order(bar.symbol, bar.timestamp, Side.BUY, 10)]
        if bar.symbol == "MSFT" and bar.timestamp.day == 2:
            return [Order(bar.symbol, bar.timestamp, Side.BUY, 5)]
        return []


def main() -> None:
    bars = [
        Bar("AAPL", datetime(2024, 1, 1, tzinfo=UTC), 100, 100, 100, 100),
        Bar("MSFT", datetime(2024, 1, 1, tzinfo=UTC), 50, 50, 50, 50),
        Bar("AAPL", datetime(2024, 1, 2, tzinfo=UTC), 110, 110, 110, 110),
        Bar("MSFT", datetime(2024, 1, 2, tzinfo=UTC), 55, 55, 55, 55),
    ]
    result = BacktestEngine().run(bars, RotationStrategy())

    print("CortexBacktest - exposure report demo")
    print("positions")
    print(result.records.positions())
    print()
    print("exposure summary")
    print(result.exposure.summary())
    print()
    print("group exposure")
    print(result.exposure.by_group({"AAPL": "technology", "MSFT": "technology"}))
    print()
    print("turnover")
    print(result.exposure.turnover())


if __name__ == "__main__":
    main()
