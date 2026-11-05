import json
from datetime import UTC, datetime

import polars as pl
import pytest

from cortexbacktest import (
    BacktestEngine,
    Bar,
    BrokerSimulator,
    EventRecordSink,
    Fill,
    Order,
    Side,
    Strategy,
    event_report,
    event_report_html,
    write_event_report_html,
    write_event_report_json,
)
from cortexbacktest.reporting import frame_records
from cortexbacktest import FillPrice


class RoundTripStrategy(Strategy):
    def on_bar(self, bar: Bar):
        if bar.timestamp.day == 1:
            return [Order(bar.symbol, bar.timestamp, Side.BUY, 10)]
        if bar.timestamp.day == 2:
            return [Order(bar.symbol, bar.timestamp, Side.SELL, 10)]
        return []


def _result():
    bars = [
        Bar("AAPL", datetime(2024, 1, 1, tzinfo=UTC), 100, 100, 100, 100),
        Bar("AAPL", datetime(2024, 1, 2, tzinfo=UTC), 110, 110, 110, 110),
    ]
    return BacktestEngine(trade_on_close=True).run(bars, RoundTripStrategy())


def test_event_report_builds_json_safe_summary_and_records():
    result = _result()

    report = event_report(result)

    assert isinstance(result.performance.equity_curve(), pl.DataFrame)
    assert report["summary"]["fills"] == 2
    assert report["counts"]["round_trips"] == 1
    assert report["trades"]["trade_count"] == 1
    assert report["pnl_reconciliation"]["reconciliation_error"] == 0
    assert report["records"]["orders"][0]["timestamp"] == "2024-01-01T00:00:00+00:00"
    assert report["records"]["equity_curve"][0]["timestamp"] == "2024-01-01T00:00:00+00:00"
    assert report["records"]["trades"][0]["holding_period"].startswith("P")
    assert report["records"]["fills"][0]["symbol"] == "AAPL"
    assert report["records"]["exposure"][0]["timestamp"] == "2024-01-01T00:00:00+00:00"
    assert report["records"]["turnover"][0]["traded_notional"] == 1000.0
    assert report["records"]["pnl_reconciliation"][0]["total_pnl"] == 100.0
    assert report["records"]["drawdowns"] == []
    json.dumps(report)


def test_backtest_result_report_can_skip_records():
    report = _result().report(include_records=False)

    assert "records" not in report
    assert report["performance"]["total_return"] > 0


def test_backtest_result_summary_retention_keeps_counts_without_records():
    bars = [
        Bar("AAPL", datetime(2024, 1, 1, tzinfo=UTC), 100, 100, 100, 100),
        Bar("AAPL", datetime(2024, 1, 2, tzinfo=UTC), 110, 110, 110, 110),
    ]

    result = BacktestEngine(trade_on_close=True, result_retention="summary").run(bars, RoundTripStrategy())
    report = result.report(include_records=False)

    assert not result.records_retained
    assert result.summary()["fills"] == 2
    assert result.summary()["order_records"] == 6
    assert report["counts"]["fills"] == 2
    assert report["counts"]["orders"] == 6
    assert report["pnl_reconciliation"]["fills"] == 0
    assert report["records_retained"] is False
    assert "records" not in report
    with pytest.raises(RuntimeError, match="event records were not retained"):
        result.records.fills()
    with pytest.raises(RuntimeError, match="event records were not retained"):
        result.report(include_records=True)


def test_summary_retention_discards_broker_history_after_notifications():
    bars = [
        Bar("AAPL", datetime(2024, 1, 1, tzinfo=UTC), 100, 100, 100, 100),
        Bar("AAPL", datetime(2024, 1, 2, tzinfo=UTC), 110, 110, 110, 110),
    ]
    broker = BrokerSimulator(fill_price=FillPrice.CLOSE)
    strategy = RoundTripStrategy()

    result = BacktestEngine(trade_on_close=True, broker=broker, result_retention="summary").run(bars, strategy)

    assert result.summary()["fills"] == 2
    assert result.summary()["order_records"] == 6
    assert broker.fill_count == 2
    assert broker.order_record_count == 6
    assert broker.fills == []
    assert broker.order_records == []


def test_event_record_sink_streams_records_without_result_retention(tmp_path):
    bars = [
        Bar("AAPL", datetime(2024, 1, 1, tzinfo=UTC), 100, 100, 100, 100),
        Bar("AAPL", datetime(2024, 1, 2, tzinfo=UTC), 110, 110, 110, 110),
    ]
    sink = EventRecordSink(tmp_path / "records", format="ndjson")

    result = BacktestEngine(trade_on_close=True, record_sink=sink, result_retention="summary").run(bars, RoundTripStrategy())

    assert result.records_retained is False
    assert result.summary()["fills"] == 2
    assert (tmp_path / "records" / "orders.ndjson").read_text().count("\n") == 6
    assert (tmp_path / "records" / "fills.ndjson").read_text().count("\n") == 2
    assert (tmp_path / "records" / "equity_curve.ndjson").exists()
    assert (tmp_path / "records" / "positions.ndjson").exists()


def test_event_record_sink_buffers_until_flush_or_close(tmp_path):
    """Nothing hits disk until the buffer fills, flush() is called, or close()."""
    sink = EventRecordSink(tmp_path / "records", format="ndjson", buffer_rows=1_000)
    fill = Fill("AAPL", datetime(2024, 1, 1, tzinfo=UTC), Side.BUY, 10, 100.0)

    sink.write_fill(fill)
    assert (tmp_path / "records" / "fills.ndjson").read_text() == ""

    sink.flush()
    assert (tmp_path / "records" / "fills.ndjson").read_text().count("\n") == 1

    sink.write_fill(fill)
    sink.close()
    assert (tmp_path / "records" / "fills.ndjson").read_text().count("\n") == 2


def test_event_record_sink_buffer_rows_auto_flushes(tmp_path):
    """The buffer flushes itself once it reaches buffer_rows, with no explicit flush()."""
    sink = EventRecordSink(tmp_path / "records", format="ndjson", buffer_rows=2)
    fill = Fill("AAPL", datetime(2024, 1, 1, tzinfo=UTC), Side.BUY, 10, 100.0)

    sink.write_fill(fill)
    assert (tmp_path / "records" / "fills.ndjson").read_text() == ""
    sink.write_fill(fill)  # buffer_rows reached: flushes without being asked
    assert (tmp_path / "records" / "fills.ndjson").read_text().count("\n") == 2
    sink.close()


def test_event_record_sink_parquet_round_trips_every_batch(tmp_path):
    """A buffer_rows of 1 forces one ParquetWriter.write_table() call per
    record, exercising the fixed record schema across successive batches."""
    bars = [
        Bar("AAPL", datetime(2024, 1, 1, tzinfo=UTC), 100, 100, 100, 100),
        Bar("AAPL", datetime(2024, 1, 2, tzinfo=UTC), 110, 110, 110, 110),
    ]
    sink = EventRecordSink(tmp_path / "records", format="parquet", buffer_rows=1)

    result = BacktestEngine(trade_on_close=True, record_sink=sink, result_retention="summary").run(bars, RoundTripStrategy())

    assert result.summary()["fills"] == 2
    orders = pl.read_parquet(tmp_path / "records" / "orders.parquet")
    fills = pl.read_parquet(tmp_path / "records" / "fills.parquet")
    assert orders.height == 6
    assert fills.height == 2
    assert (tmp_path / "records" / "equity_curve.parquet").exists()
    # RoundTripStrategy buys then fully sells, so it ends flat: an empty but
    # schema-valid Parquet file, not a missing one.
    positions = pl.read_parquet(tmp_path / "records" / "positions.parquet")
    assert positions.height == 0
    assert positions.columns == ["symbol", "quantity", "last_price", "market_value"]


def test_frame_records_accepts_polars_tables():
    frame = pl.DataFrame(
        {
            "timestamp": [datetime(2024, 1, 1, tzinfo=UTC)],
            "duration": [datetime(2024, 1, 2, tzinfo=UTC) - datetime(2024, 1, 1, tzinfo=UTC)],
            "value": [1.5],
        }
    )

    records = frame_records(frame)

    assert records[0]["timestamp"] == "2024-01-01T00:00:00+00:00"
    assert records[0]["duration"].startswith("P1D")
    assert records[0]["value"] == 1.5


def test_frame_records_preserves_nested_temporal_and_nonfinite_values():
    from datetime import date, time, timedelta

    frame = pl.DataFrame({
        "day": [date(2024, 1, 1)], "time": [time(12, 30)],
        "duration": [timedelta(days=-1, seconds=-2)],
        "value": [float("inf")], "missing": [None], "count": [2**60], "flag": [True],
        "nested": [{"values": [1.0, float("nan")], "day": date(2024, 1, 2)}],
    })
    records = frame_records(frame)
    assert records == [{"day": "2024-01-01", "time": "12:30:00", "duration": "-P1DT2S",
                        "value": None, "missing": None, "count": 2**60, "flag": True,
                        "nested": {"values": [1.0, None], "day": "2024-01-02"}}]
    json.dumps(records, allow_nan=False)
    assert frame_records(pl.Series("", [float("nan"), 1.0])) == [{"value": None}, {"value": 1.0}]


def test_report_does_not_reuse_trade_analysis_after_result_mutation():
    result = _result()
    original = result.report()
    assert original["counts"]["round_trips"] > 0
    result.fills.clear()
    updated = result.report()
    assert updated["counts"]["round_trips"] == 0
    assert updated["records"]["trades"] == updated["records"]["fills"] == []
    assert updated["pnl_reconciliation"]["fills"] == 0


def test_write_event_report_json(tmp_path):
    output_path = write_event_report_json(_result(), tmp_path / "report.json")

    payload = json.loads(output_path.read_text())

    assert payload["summary"]["fills"] == 2
    assert payload["records"]["fills"][0]["order_id"] == "O00000001"


def test_event_report_html_renders_standalone_document():
    html = event_report_html(_result())

    assert "<!doctype html>" in html
    assert "CortexBacktest Report" in html
    assert "Final Equity" in html
    assert "Round Trips" in html
    assert "AAPL" in html


def test_write_event_report_html(tmp_path):
    output_path = write_event_report_html(_result(), tmp_path / "report.html")

    html = output_path.read_text()

    assert "<html" in html
    assert "Trade Statistics" in html
    assert "PnL Reconciliation" in html
