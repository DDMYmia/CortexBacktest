"""Reject concentrated orders with portfolio-level risk rules."""

from __future__ import annotations

from datetime import UTC, datetime

from cortexbacktest import BacktestEngine, Bar, MaxPositionWeightRule, Order, Side, Strategy
from cortexbacktest.broker import BrokerSimulator


class ConcentratedOrderStrategy(Strategy):
    def __init__(self) -> None:
        self.submitted = False

    def on_bar(self, bar: Bar) -> list[Order]:
        if self.submitted:
            return []
        self.submitted = True
        return [Order(bar.symbol, bar.timestamp, Side.BUY, 8)]


def main() -> None:
    bars = [Bar("AAPL", datetime(2024, 1, 1, tzinfo=UTC), 100, 100, 100, 100)]
    broker = BrokerSimulator(initial_cash=1_000, risk_rules=[MaxPositionWeightRule(max_weight=0.5)])
    result = BacktestEngine(broker=broker).run(bars, ConcentratedOrderStrategy())

    print("CortexBacktest - portfolio risk demo")
    print(result.records.orders().select(["symbol", "quantity", "status", "reason"]))
    print(f"fills: {len(result.fills)}")


if __name__ == "__main__":
    main()
