"""Inspect open orders and cancellation lifecycle."""

from __future__ import annotations

from datetime import UTC, datetime

from cortexbacktest import Bar, Order, OrderType, Side
from cortexbacktest.broker import BrokerSimulator


def main() -> None:
    broker = BrokerSimulator()
    timestamp = datetime(2024, 1, 1, tzinfo=UTC)
    submitted = broker.submit_order(
        Order(
            symbol="aapl",
            timestamp=timestamp,
            side=Side.BUY,
            quantity=10,
            order_type=OrderType.LIMIT,
            limit_price=95.0,
            tag="resting-limit",
        )
    )
    broker.process_bar(Bar("AAPL", timestamp, 100, 101, 99, 100))
    cancelled = broker.cancel_order(submitted.order.id, reason="strategy replaced order")

    print("CortexBacktest - order lifecycle demo")
    print(f"submitted_id: {submitted.order.id}")
    print(f"submitted_status: {submitted.status.value}")
    print(f"cancelled_status: {cancelled.status.value if cancelled else None}")
    print(f"open_orders: {len(broker.open_orders)}")
    print(f"fills: {len(broker.fills)}")


if __name__ == "__main__":
    main()
