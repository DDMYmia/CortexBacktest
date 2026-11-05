from datetime import UTC, datetime

import polars as pl

from cortexbacktest import BacktestEngine, Bar, Order, Side, Strategy
from cortexbacktest.event import BrokerSimulator
from cortexbacktest.records import (
    fills_frame,
    open_orders_frame,
    order_records_frame,
    write_fills_ndjson,
    write_order_records_csv,
)
from cortexbacktest.reporting import trade_summary
from cortexbacktest import FillPrice


class BuyOnceStrategy(Strategy):
    def __init__(self) -> None:
        self.ordered = False

    def on_bar(self, bar: Bar):
        if self.ordered:
            return []
        self.ordered = True
        return [Order(bar.symbol, bar.timestamp, Side.BUY, 2)]


def _result():
    bars = [
        Bar("AAPL", datetime(2024, 1, 1, tzinfo=UTC), 10, 10, 10, 10),
        Bar("AAPL", datetime(2024, 1, 2, tzinfo=UTC), 11, 11, 11, 11),
    ]
    return BacktestEngine(trade_on_close=True).run(bars, BuyOnceStrategy())


def test_backtest_result_exposes_polars_record_frames():
    result = _result()

    orders = result.records.orders()
    fills = result.records.fills()
    positions = result.records.positions()

    assert isinstance(orders, pl.DataFrame)
    assert isinstance(fills, pl.DataFrame)
    assert isinstance(positions, pl.DataFrame)
    assert orders["status"].to_list() == result.records.orders()["status"].to_list()
    assert fills["symbol"].to_list() == result.records.fills()["symbol"].to_list()
    assert positions["symbol"].to_list() == result.records.positions()["symbol"].to_list()
    assert result.trades.fill_summary() == trade_summary(fills)
    assert result.trades.fill_summary() == trade_summary(result.records.fills())


def test_backtest_result_keeps_equity_curve_in_polars_storage():
    result = _result()

    assert isinstance(result._equity_curve_pl, pl.DataFrame)
    assert isinstance(result.performance.equity_curve(), pl.DataFrame)
    assert result.performance.equity_curve().height == result._equity_curve_pl.height


def test_top_level_polars_record_builders_handle_empty_inputs():
    assert order_records_frame([]).columns == [
        "timestamp",
        "order_id",
        "symbol",
        "side",
        "quantity",
        "order_type",
        "limit_price",
        "stop_price",
        "trail_amount",
        "trail_percent",
        "expires_at",
        "time_in_force",
        "tag",
        "oco_group",
        "parent_id",
        "status",
        "status_label",
        "reason",
    ]
    assert fills_frame([]).columns == [
        "timestamp",
        "order_id",
        "symbol",
        "side",
        "quantity",
        "price",
        "commission",
        "slippage",
        "notional",
    ]
    assert open_orders_frame([]).columns == [
        "timestamp",
        "order_id",
        "symbol",
        "side",
        "quantity",
        "order_type",
        "limit_price",
        "stop_price",
        "trail_amount",
        "trail_percent",
        "expires_at",
        "time_in_force",
        "tag",
        "oco_group",
        "parent_id",
    ]


class LargeOrderStrategy(Strategy):
    def __init__(self) -> None:
        self.ordered = False

    def on_bar(self, bar: Bar):
        if self.ordered:
            return []
        self.ordered = True
        return [Order(bar.symbol, bar.timestamp, Side.BUY, 80)]


def test_order_records_pl_includes_backtrader_style_status_label():
    bars = [
        Bar("AAPL", datetime(2024, 1, 1, tzinfo=UTC), 10, 10, 10, 10, volume=100),
        Bar("AAPL", datetime(2024, 1, 2, tzinfo=UTC), 10, 10, 10, 10, volume=100),
    ]
    result = BacktestEngine(trade_on_close=True, broker=BrokerSimulator(fill_price=FillPrice.CLOSE, max_volume_share=0.5)).run(
        bars,
        LargeOrderStrategy(),
    )

    orders = result.records.orders()

    assert orders["status"].to_list() == ["submitted", "accepted", "partially_filled", "filled"]
    assert orders["status_label"].to_list() == ["submitted", "accepted", "partial", "filled"]


def test_streaming_record_writers_and_result_directory_export(tmp_path):
    result = _result()

    orders_csv = write_order_records_csv(result.orders, tmp_path / "orders.csv")
    fills_ndjson = write_fills_ndjson(result.fills, tmp_path / "fills.ndjson")
    records_dir = result.records.write_dir(tmp_path / "records", format="csv")

    assert orders_csv.read_text().splitlines()[0].startswith("timestamp,order_id,symbol")
    assert '"symbol": "AAPL"' in fills_ndjson.read_text()
    assert (records_dir / "equity_curve.csv").exists()
    assert (records_dir / "orders.csv").exists()
    assert (records_dir / "fills.csv").exists()


def test_result_directory_export_supports_ndjson(tmp_path):
    result = _result()

    records_dir = result.records.write_dir(tmp_path / "records_ndjson", format="ndjson")

    assert (records_dir / "equity_curve.ndjson").exists()
    assert (records_dir / "orders.ndjson").exists()
    assert (records_dir / "fills.ndjson").exists()
