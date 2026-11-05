"""Reject sector-concentrated orders with group exposure rules."""

from __future__ import annotations

from datetime import UTC, datetime

from cortexbacktest import BacktestEngine, Bar, MaxGroupWeightRule, Order, Side, Strategy
from cortexbacktest.broker import BrokerSimulator
from cortexbacktest.contracts import Position


class TechnologyOrderStrategy(Strategy):
    def __init__(self) -> None:
        self.submitted = False

    def on_bar(self, bar: Bar) -> list[Order]:
        if self.submitted:
            return []
        self.submitted = True
        return [Order("AAPL", bar.timestamp, Side.BUY, 5)]


def main() -> None:
    bars = [Bar("AAPL", datetime(2024, 1, 1, tzinfo=UTC), 100, 100, 100, 100)]
    broker = BrokerSimulator(
        initial_cash=1_000,
        risk_rules=[
            MaxGroupWeightRule(
                groups={"AAPL": "technology", "MSFT": "technology", "XOM": "energy"},
                max_weights={"technology": 0.5},
            )
        ],
    )
    broker.state.positions["MSFT"] = Position("MSFT", 2, 100)
    result = BacktestEngine(broker=broker).run(bars, TechnologyOrderStrategy())

    print("CortexBacktest - group risk demo")
    print(result.records.orders().select(["symbol", "quantity", "status", "reason"]))
    print(f"fills: {len(result.fills)}")


if __name__ == "__main__":
    main()
