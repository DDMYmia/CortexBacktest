from datetime import UTC, datetime

from cortexbacktest import BacktestEngine, Bar, Order, OrderStatus, OrderType, Side, Strategy
from cortexbacktest.event import BrokerSimulator
from cortexbacktest import FillPrice


class NotificationStrategy(Strategy):
    def __init__(self, orders_by_day) -> None:
        self.orders_by_day = orders_by_day
        self.notifications = []

    def notify_order(self, record):
        self.notifications.append((record.order.id, record.order.symbol, record.status, record.reason))

    def on_bar(self, bar: Bar):
        return self.orders_by_day.get(bar.timestamp.day, [])


class NoNotificationStrategy(Strategy):
    def on_bar(self, bar: Bar):
        if bar.timestamp.day == 1:
            return [Order("AAPL", bar.timestamp, Side.BUY, 1)]
        return []


def test_strategy_notified_for_submitted_accepted_and_filled():
    bars = [Bar("AAPL", datetime(2024, 1, 1, tzinfo=UTC), 10, 10, 10, 10)]
    strategy = NotificationStrategy(
        {1: [Order("AAPL", bars[0].timestamp, Side.BUY, 2)]}
    )

    BacktestEngine(trade_on_close=True).run(bars, strategy)

    assert [status for _, _, status, _ in strategy.notifications] == [
        OrderStatus.SUBMITTED,
        OrderStatus.ACCEPTED,
        OrderStatus.FILLED,
    ]


def test_default_noop_order_notification_uses_fast_path():
    bars = [Bar("AAPL", datetime(2024, 1, 1, tzinfo=UTC), 10, 10, 10, 10)]
    broker = BrokerSimulator(fill_price=FillPrice.CLOSE)

    result = BacktestEngine(trade_on_close=True, broker=broker, result_retention="summary").run(bars, NoNotificationStrategy())

    assert result.summary()["order_records"] == 3
    assert broker.order_manager.pending_records == []
    assert broker.order_records == []
    assert broker.order_record_count == 3


def test_summary_retention_still_notifies_strategy_order_records():
    bars = [Bar("AAPL", datetime(2024, 1, 1, tzinfo=UTC), 10, 10, 10, 10)]
    strategy = NotificationStrategy(
        {1: [Order("AAPL", bars[0].timestamp, Side.BUY, 2)]}
    )
    broker = BrokerSimulator(fill_price=FillPrice.CLOSE)

    BacktestEngine(trade_on_close=True, broker=broker, result_retention="summary").run(bars, strategy)

    assert [status for _, _, status, _ in strategy.notifications] == [
        OrderStatus.SUBMITTED,
        OrderStatus.ACCEPTED,
        OrderStatus.FILLED,
    ]
    assert broker.order_records == []
    assert broker.order_record_count == 3


def test_strategy_notified_for_rejected_order():
    bars = [Bar("AAPL", datetime(2024, 1, 1, tzinfo=UTC), 10, 10, 10, 10)]
    strategy = NotificationStrategy(
        {1: [Order("AAPL", bars[0].timestamp, Side.BUY, -1)]}
    )

    BacktestEngine(trade_on_close=True).run(bars, strategy)

    assert [status for _, _, status, _ in strategy.notifications] == [OrderStatus.REJECTED]
    assert strategy.notifications[0][3] == "order quantity must be positive"


def test_strategy_notified_for_partial_and_final_fill():
    bars = [
        Bar("AAPL", datetime(2024, 1, 1, tzinfo=UTC), 10, 10, 10, 10, volume=100),
        Bar("AAPL", datetime(2024, 1, 2, tzinfo=UTC), 10, 10, 10, 10, volume=100),
    ]
    strategy = NotificationStrategy(
        {1: [Order("AAPL", bars[0].timestamp, Side.BUY, 80)]}
    )
    broker = BrokerSimulator(fill_price=FillPrice.CLOSE, max_volume_share=0.5)

    BacktestEngine(trade_on_close=True, broker=broker).run(bars, strategy)

    assert [status for _, _, status, _ in strategy.notifications] == [
        OrderStatus.SUBMITTED,
        OrderStatus.ACCEPTED,
        OrderStatus.PARTIALLY_FILLED,
        OrderStatus.FILLED,
    ]


def test_order_record_exposes_backtrader_style_status_label():
    bars = [
        Bar("AAPL", datetime(2024, 1, 1, tzinfo=UTC), 10, 10, 10, 10, volume=100),
        Bar("AAPL", datetime(2024, 1, 2, tzinfo=UTC), 10, 10, 10, 10, volume=100),
    ]
    strategy = NotificationStrategy(
        {1: [Order("AAPL", bars[0].timestamp, Side.BUY, 80)]}
    )
    broker = BrokerSimulator(fill_price=FillPrice.CLOSE, max_volume_share=0.5)

    result = BacktestEngine(trade_on_close=True, broker=broker).run(bars, strategy)

    assert [record.status_label for record in result.orders] == [
        "submitted",
        "accepted",
        "partial",
        "filled",
    ]


class CancelNotificationStrategy(Strategy):
    def __init__(self) -> None:
        self.order_id = None
        self.notifications = []

    def notify_order(self, record):
        self.notifications.append(record.status)
        if record.status == OrderStatus.ACCEPTED:
            self.order_id = record.order.id

    def on_bar(self, bar: Bar):
        if bar.timestamp.day == 1:
            return [
                Order(
                    "AAPL",
                    bar.timestamp,
                    Side.BUY,
                    1,
                    order_type=OrderType.LIMIT,
                    limit_price=1.0,
                )
            ]
        if bar.timestamp.day == 2 and self.order_id is not None:
            self.broker.cancel_order(self.order_id, reason="strategy cancel")
        return []


def test_strategy_notified_for_cancelled_order():
    bars = [
        Bar("AAPL", datetime(2024, 1, 1, tzinfo=UTC), 10, 10, 10, 10),
        Bar("AAPL", datetime(2024, 1, 2, tzinfo=UTC), 10, 10, 10, 10),
    ]
    strategy = CancelNotificationStrategy()

    BacktestEngine(trade_on_close=False).run(bars, strategy)

    assert strategy.notifications == [
        OrderStatus.SUBMITTED,
        OrderStatus.ACCEPTED,
        OrderStatus.CANCELLED,
    ]


class ReplaceNotificationStrategy(Strategy):
    def __init__(self) -> None:
        self.order_id = None
        self.notifications = []

    def notify_order(self, record):
        self.notifications.append((record.status, record.order.limit_price, record.order.quantity))
        if record.status == OrderStatus.ACCEPTED:
            self.order_id = record.order.id

    def on_bar(self, bar: Bar):
        if bar.timestamp.day == 1:
            return [
                Order(
                    "AAPL",
                    bar.timestamp,
                    Side.BUY,
                    1,
                    order_type=OrderType.LIMIT,
                    limit_price=1.0,
                )
            ]
        if bar.timestamp.day == 2 and self.order_id is not None:
            self.replace_order(self.order_id, limit_price=10.0, quantity=2)
        return []


def test_strategy_notified_for_replaced_order_and_fill():
    bars = [
        Bar("AAPL", datetime(2024, 1, 1, tzinfo=UTC), 10, 10, 10, 10),
        Bar("AAPL", datetime(2024, 1, 2, tzinfo=UTC), 10, 10, 10, 10),
        Bar("AAPL", datetime(2024, 1, 3, tzinfo=UTC), 10, 10, 10, 10),
    ]
    strategy = ReplaceNotificationStrategy()

    BacktestEngine(trade_on_close=False).run(bars, strategy)

    assert [status for status, _, _ in strategy.notifications] == [
        OrderStatus.SUBMITTED,
        OrderStatus.ACCEPTED,
        OrderStatus.REPLACED,
        OrderStatus.FILLED,
    ]
    assert strategy.notifications[2][1:] == (10.0, 2)


def test_summary_retention_notifies_replaced_order_without_retaining_records():
    bars = [
        Bar("AAPL", datetime(2024, 1, 1, tzinfo=UTC), 10, 10, 10, 10),
        Bar("AAPL", datetime(2024, 1, 2, tzinfo=UTC), 10, 10, 10, 10),
        Bar("AAPL", datetime(2024, 1, 3, tzinfo=UTC), 10, 10, 10, 10),
    ]
    strategy = ReplaceNotificationStrategy()
    broker = BrokerSimulator(fill_price=FillPrice.CLOSE)

    result = BacktestEngine(broker=broker, trade_on_close=False, result_retention="summary").run(bars, strategy)

    assert [status for status, _, _ in strategy.notifications] == [
        OrderStatus.SUBMITTED,
        OrderStatus.ACCEPTED,
        OrderStatus.REPLACED,
        OrderStatus.FILLED,
    ]
    assert result.summary()["order_records"] == 4
    assert broker.order_records == []


def test_strategy_notified_for_expired_order():
    bars = [
        Bar("AAPL", datetime(2024, 1, 1, tzinfo=UTC), 10, 10, 10, 10),
        Bar("AAPL", datetime(2024, 1, 2, tzinfo=UTC), 10, 10, 10, 10),
    ]
    strategy = NotificationStrategy(
        {
            1: [
                Order(
                    "AAPL",
                    bars[0].timestamp,
                    Side.BUY,
                    1,
                    order_type=OrderType.LIMIT,
                    limit_price=1.0,
                    expires_at=bars[0].timestamp,
                )
            ]
        }
    )

    BacktestEngine(trade_on_close=False).run(bars, strategy)

    assert [status for _, _, status, _ in strategy.notifications] == [
        OrderStatus.SUBMITTED,
        OrderStatus.ACCEPTED,
        OrderStatus.EXPIRED,
    ]
