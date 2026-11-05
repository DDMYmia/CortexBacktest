"""Run a volume-limited partial-fill backtest."""

from __future__ import annotations

from datetime import UTC, datetime

from cortexbacktest import BacktestEngine, Bar, Order, Side, Strategy
from cortexbacktest.broker import BrokerSimulator


class LargeOrderStrategy(Strategy):
    def __init__(self) -> None:
        self.submitted = False

    def on_bar(self, bar: Bar) -> list[Order]:
        if self.submitted:
            return []
        self.submitted = True
        return [Order(symbol=bar.symbol, timestamp=bar.timestamp, side=Side.BUY, quantity=80)]


def main() -> None:
    bars = [
        Bar("AAPL", datetime(2024, 1, 1, tzinfo=UTC), 10, 10, 10, 10, volume=100),
        Bar("AAPL", datetime(2024, 1, 2, tzinfo=UTC), 10, 10, 10, 10, volume=100),
    ]
    broker = BrokerSimulator(max_volume_share=0.5)
    result = BacktestEngine(broker=broker).run(bars, LargeOrderStrategy())

    print("CortexBacktest - partial fill demo")
    print("orders:")
    print(result.records.orders().select(["order_id", "status", "quantity", "reason"]))
    print("fills:")
    print(result.records.fills().select(["order_id", "quantity", "price", "notional"]))
    print(f"open_orders: {len(result.open_orders)}")


if __name__ == "__main__":
    main()
