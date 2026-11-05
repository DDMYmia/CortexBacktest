from datetime import UTC, datetime

from cortexbacktest import (
    BacktestEngine,
    Bar,
    EventAnalyzer,
    Order,
    OrderFlowAnalyzer,
    Side,
    Strategy,
)
from cortexbacktest.event import BrokerSimulator
from cortexbacktest import FillPrice


class BuyOnceStrategy(Strategy):
    def __init__(self) -> None:
        self.bought = False

    def on_bar(self, bar: Bar) -> list[Order]:
        if self.bought:
            return []
        self.bought = True
        return [Order(bar.symbol, bar.timestamp, Side.BUY, 10)]


class TraceAnalyzer(EventAnalyzer):
    name = "trace"

    def __init__(self) -> None:
        self.events = []

    def on_start(self, *, broker, strategy) -> None:
        self.events.append("start")

    def before_bar(self, bar: Bar, *, broker, strategy) -> None:
        self.events.append(f"before:{bar.timestamp.date()}")

    def on_order(self, order: Order, record, *, broker, strategy) -> None:
        self.events.append(f"order:{record.status.value}")

    def on_fill(self, fill, *, broker, strategy) -> None:
        self.events.append(f"fill:{fill.symbol}")

    def after_bar(self, bar: Bar, *, broker, strategy) -> None:
        self.events.append(f"after:{bar.timestamp.date()}")

    def on_finish(self, *, broker, strategy):
        self.events.append("finish")
        return {"events": list(self.events)}


def _bars():
    return [
        Bar("AAPL", datetime(2024, 1, 1, tzinfo=UTC), 10, 10, 10, 10),
        Bar("AAPL", datetime(2024, 1, 2, tzinfo=UTC), 12, 12, 12, 12),
    ]


def test_event_analyzer_receives_lifecycle_hooks():
    analyzer = TraceAnalyzer()

    result = BacktestEngine(trade_on_close=True, analyzers=[analyzer]).run(_bars(), BuyOnceStrategy())

    assert result.analyzer_results == {
        "trace": {
            "events": [
                "start",
                "before:2024-01-01",
                "order:submitted",
                "fill:AAPL",
                "after:2024-01-01",
                "before:2024-01-02",
                "after:2024-01-02",
                "finish",
            ]
        }
    }


def test_order_flow_analyzer_summarizes_orders_and_fills():
    result = BacktestEngine(trade_on_close=True, analyzers=[OrderFlowAnalyzer()]).run(_bars(), BuyOnceStrategy())

    assert result.analyzer_results["order_flow"]["bars"] == 2
    assert result.analyzer_results["order_flow"]["submitted"] == 1
    assert result.analyzer_results["order_flow"]["accepted"] == 1
    assert result.analyzer_results["order_flow"]["filled"] == 1
    assert result.analyzer_results["order_flow"]["fills"] == 1
    assert result.analyzer_results["order_flow"]["gross_notional"] == 100.0


def test_summary_retention_keeps_broker_history_when_analyzers_need_it():
    broker = BrokerSimulator(fill_price=FillPrice.CLOSE)
    result = BacktestEngine(trade_on_close=True, 
        broker=broker,
        analyzers=[OrderFlowAnalyzer()],
        result_retention="summary",
    ).run(_bars(), BuyOnceStrategy())

    assert result.records_retained is False
    assert result.analyzer_results["order_flow"]["filled"] == 1
    assert broker.order_record_count == 3
    assert len(broker.order_records) == 3
    assert broker.fill_count == 1
    assert len(broker.fills) == 1


class LargeOrderStrategy(Strategy):
    def __init__(self) -> None:
        self.ordered = False

    def on_bar(self, bar: Bar) -> list[Order]:
        if self.ordered:
            return []
        self.ordered = True
        return [Order(bar.symbol, bar.timestamp, Side.BUY, 80)]


def test_order_flow_analyzer_reports_partial_alias():
    bars = [
        Bar("AAPL", datetime(2024, 1, 1, tzinfo=UTC), 10, 10, 10, 10, volume=100),
        Bar("AAPL", datetime(2024, 1, 2, tzinfo=UTC), 10, 10, 10, 10, volume=100),
    ]
    result = BacktestEngine(trade_on_close=True, 
        broker=BrokerSimulator(fill_price=FillPrice.CLOSE, max_volume_share=0.5),
        analyzers=[OrderFlowAnalyzer()],
    ).run(bars, LargeOrderStrategy())

    assert result.analyzer_results["order_flow"]["partial"] == 1
    assert result.analyzer_results["order_flow"]["partially_filled"] == 1
