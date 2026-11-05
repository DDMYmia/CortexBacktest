"""Demonstrate stop and stop-limit order handling."""

from __future__ import annotations

from datetime import UTC, datetime

from cortexbacktest import Bar, Order, OrderType, Side
from cortexbacktest.broker import BrokerSimulator


def main() -> None:
    timestamp = datetime(2024, 1, 1, tzinfo=UTC)
    broker = BrokerSimulator(allow_short=True)
    broker.submit_order(
        Order(
            symbol="AAPL",
            timestamp=timestamp,
            side=Side.BUY,
            quantity=10,
            order_type=OrderType.STOP,
            stop_price=105.0,
            tag="breakout-stop",
        )
    )
    broker.submit_order(
        Order(
            symbol="AAPL",
            timestamp=timestamp,
            side=Side.SELL,
            quantity=5,
            order_type=OrderType.STOP_LIMIT,
            stop_price=95.0,
            limit_price=94.5,
            tag="protective-stop-limit",
        )
    )

    first_bar = Bar("AAPL", timestamp, open=100, high=103, low=98, close=101)
    second_bar = Bar("AAPL", datetime(2024, 1, 2, tzinfo=UTC), open=101, high=106, low=94, close=96)
    broker.process_bar(first_bar)
    broker.process_bar(second_bar)

    print("CortexBacktest - stop orders demo")
    for fill in broker.fills:
        print(f"{fill.side.value} {fill.quantity:g} {fill.symbol} @ {fill.price:.2f} order={fill.order_id}")
    print(f"open_orders: {len(broker.open_orders)}")


if __name__ == "__main__":
    main()
