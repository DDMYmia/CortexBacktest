"""Compare supported event-engine fill price models."""

from __future__ import annotations

from datetime import UTC, datetime

from cortexbacktest import Bar, FillPrice, Order, Side
from cortexbacktest.broker import BrokerSimulator


def main() -> None:
    timestamp = datetime(2024, 1, 1, tzinfo=UTC)
    bar = Bar("AAPL", timestamp, open=10, high=16, low=8, close=14, volume=1_000, vwap=12.5)

    print("CortexBacktest - fill price models demo")
    for model in (FillPrice.OPEN, FillPrice.CLOSE, FillPrice.MID, FillPrice.HLC3, FillPrice.OHLC4, FillPrice.VWAP):
        broker = BrokerSimulator(fill_price=model)
        fill = broker.execute(Order("AAPL", timestamp, Side.BUY, 1), bar)
        print(f"{model.value}: {fill.price:.4f}")


if __name__ == "__main__":
    main()
