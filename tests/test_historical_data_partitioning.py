"""HistoricalDataStore's partitioned tick/order-book layout.

Covers what test_backtesting_vnpy_compat.py's and test_order_book_history.py's
single-flush, single-date fixtures do not exercise: multiple flushes across
multiple dates, date-range pruning and lazy scanning.
"""

from datetime import UTC, date, datetime

import polars as pl

from cortexbacktest import (
    HistoricalDataStore,
    OrderBookEvent,
    OrderBookRecorder,
    Side,
    Tick,
    TickRecorder,
)


def _tick(day: int, hour: int, price: float) -> Tick:
    return Tick("AAPL", datetime(2024, 1, day, hour, 0, tzinfo=UTC), price)


def test_append_never_rewrites_a_prior_flushs_file(tmp_path):
    """Each flush adds a part file; it never touches one written by an earlier flush."""
    store = HistoricalDataStore(tmp_path)
    recorder = TickRecorder(store, "demo")

    recorder.record(_tick(1, 9, 10.0))
    recorder.flush()
    first_parts = set((tmp_path / "ticks" / "demo").glob("date=*/*.parquet"))

    recorder.record(_tick(1, 10, 11.0))
    recorder.flush()
    second_parts = set((tmp_path / "ticks" / "demo").glob("date=*/*.parquet"))

    assert first_parts < second_parts  # strictly grew; nothing already there was replaced
    assert all(path.exists() for path in first_parts)  # and the originals are untouched
    # read_ticks makes no ordering promise across part files; TickReplayer is
    # what sorts (see test_historical_data_store_and_tick_replayer).
    assert sorted(tick.last_price for tick in store.read_ticks("demo")) == [10.0, 11.0]


def test_a_single_flush_spanning_dates_splits_into_one_partition_per_date(tmp_path):
    store = HistoricalDataStore(tmp_path)
    recorder = TickRecorder(store, "multi_day")
    for day in (1, 2, 3):
        recorder.record(_tick(day, 9, float(day)))
    recorder.flush()

    dataset_dir = tmp_path / "ticks" / "multi_day"
    assert sorted(p.name for p in dataset_dir.iterdir() if p.is_dir()) == [
        "date=2024-01-01",
        "date=2024-01-02",
        "date=2024-01-03",
    ]
    assert sorted(tick.timestamp.day for tick in store.read_ticks("multi_day")) == [1, 2, 3]


def test_read_ticks_date_range_prunes_partitions(tmp_path):
    store = HistoricalDataStore(tmp_path)
    recorder = TickRecorder(store, "ranged")
    for day in (1, 2, 3, 4, 5):
        recorder.record(_tick(day, 9, float(day)))
    recorder.flush()

    ticks = store.read_ticks("ranged", start_date=date(2024, 1, 2), end_date=date(2024, 1, 4))

    assert sorted(tick.timestamp.day for tick in ticks) == [2, 3, 4]


def test_scan_ticks_returns_a_lazyframe_with_the_raw_columns(tmp_path):
    store = HistoricalDataStore(tmp_path)
    recorder = TickRecorder(store, "scanned")
    recorder.record(_tick(1, 9, 10.0))
    recorder.record(_tick(2, 9, 12.0))
    recorder.flush()

    lazy = store.scan_ticks("scanned", start_date=date(2024, 1, 2))
    assert isinstance(lazy, pl.LazyFrame)
    frame = lazy.collect()
    assert frame["last_price"].to_list() == [12.0]
    assert "date" not in frame.columns  # the synthetic hive-partition column is dropped


def test_write_ticks_and_write_order_book_events_are_partitioned_from_the_start(tmp_path):
    store = HistoricalDataStore(tmp_path)
    store.write_ticks("bulk", [_tick(1, 9, 10.0), _tick(2, 9, 11.0)])

    assert not (tmp_path / "ticks" / "bulk.parquet").exists()
    assert (tmp_path / "ticks" / "bulk" / "date=2024-01-01").exists()
    assert (tmp_path / "ticks" / "bulk" / "date=2024-01-02").exists()
    assert store.list_datasets("ticks") == ["bulk"]

    events = [OrderBookEvent("AAPL", datetime(2024, 1, 1, 9, tzinfo=UTC), Side.BUY, price=1.0, quantity=1)]
    store.write_order_book_events("l2_bulk", events)
    assert (tmp_path / "order_books" / "l2_bulk" / "date=2024-01-01").exists()


def test_write_ticks_called_twice_replaces_rather_than_appends(tmp_path):
    """write_ticks (unlike TickRecorder.flush) is a from-scratch write: the
    second call's rows entirely replace the first's, matching the pre-
    partitioning behavior."""
    store = HistoricalDataStore(tmp_path)
    store.write_ticks("overwritten", [_tick(1, 9, 10.0)])
    store.write_ticks("overwritten", [_tick(2, 9, 20.0)])

    ticks = store.read_ticks("overwritten")
    assert [tick.last_price for tick in ticks] == [20.0]


def test_list_datasets_reports_both_layouts(tmp_path):
    store = HistoricalDataStore(tmp_path)
    store.write_bars("legacy_bars", [])  # single-file kind, untouched by partitioning
    store.write_ticks("partitioned", [_tick(1, 9, 10.0)])

    assert store.list_datasets("ticks") == ["partitioned"]


def test_order_book_recorder_also_partitions_by_date(tmp_path):
    store = HistoricalDataStore(tmp_path)
    recorder = OrderBookRecorder(store, "l2")
    recorder.record(OrderBookEvent("AAPL", datetime(2024, 1, 1, 9, tzinfo=UTC), Side.BUY, price=1.0, quantity=1))
    recorder.flush()
    recorder.record(OrderBookEvent("AAPL", datetime(2024, 1, 2, 9, tzinfo=UTC), Side.BUY, price=2.0, quantity=1))
    recorder.flush()

    events = store.read_order_book_events("l2")
    assert sorted(event.timestamp.day for event in events) == [1, 2]
    assert store.list_datasets("order_books") == ["l2"]
