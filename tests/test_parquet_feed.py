from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from cortexbacktest import BacktestEngine, ColumnarBarFeed, ParquetBarFeed
from tests.test_event_feed import BuyOnceStrategy


def frame():
    return pl.DataFrame({
        "date": [datetime(2024, 1, 1, tzinfo=UTC) + timedelta(minutes=i // 3) for i in range(12)],
        "symbol": ["c", "a", "b"] * 4,
        **{name: [10.0 + i // 3 for i in range(12)] for name in ("open", "high", "low", "close")},
        "volume": [100.0] * 12,
        "open_interest": [2.0] * 12,
        "halted": [False] * 12,
        "limit_up": [20.0] * 12,
        "limit_down": [1.0] * 12,
    })


@pytest.mark.parametrize("batch_rows", [1, 2, 3, 5, 100])
def test_boundaries_replay_and_manifest_match(tmp_path, batch_rows):
    data = frame()
    paths = [tmp_path / f'{i}.parquet' for i in range(3)]
    for path, part in zip(paths, [data[:2], data[2:7], data[7:]], strict=True):
        part.write_parquet(path, row_group_size=2)
    feed = ParquetBarFeed(paths, batch_rows=batch_rows)
    expected = ColumnarBarFeed.from_polars(data)
    assert list(feed) == list(expected)
    assert list(feed) == list(expected)
    assert feed.data_fingerprint() == expected.data_fingerprint()
    actual = BacktestEngine().run(feed, BuyOnceStrategy())
    reference = BacktestEngine().run(expected, BuyOnceStrategy())
    assert actual.summary() == reference.summary()
    assert actual.fills == reference.fills
    assert actual.performance.equity_curve().equals(reference.performance.equity_curve())


@pytest.mark.parametrize("kind", ["within", "across", "duplicate", "null", "invalid"])
def test_reject_bad_data(tmp_path, kind):
    data = frame()
    if kind in {"within", "across"}:
        data = data.reverse()
    elif kind == "duplicate":
        data = data.with_columns(pl.lit("a").alias("symbol"))
    elif kind == "null":
        data = data.with_columns(pl.lit(None).cast(pl.Datetime("us", "UTC")).alias("date"))
    else:
        data = data.with_columns(pl.lit(-1.0).alias("volume"))
    path = tmp_path / 'data.parquet'
    data.write_parquet(path)
    with pytest.raises(ValueError):
        list(ParquetBarFeed(path, batch_rows=1 if kind == "across" else 100))


def test_empty_and_explicit_materialization_guard(tmp_path):
    path = tmp_path / 'empty.parquet'
    frame().head(0).write_parquet(path)
    feed = ParquetBarFeed(path)
    assert list(feed) == []
    with pytest.raises(TypeError):
        len(feed)
    with pytest.raises(TypeError):
        _ = feed.bars
    with pytest.raises(ValueError):
        ParquetBarFeed(path, batch_rows=0)


def test_file_order_rejected(tmp_path):
    paths = [tmp_path / 'late.parquet', tmp_path / 'early.parquet']
    frame()[6:].write_parquet(paths[0])
    frame()[:6].write_parquet(paths[1])
    with pytest.raises(ValueError, match="decrease"):
        list(ParquetBarFeed(paths))


@pytest.mark.parametrize('empty', [False, True])
def test_stream_validation_is_one_pass_with_identical_manifest(tmp_path, empty):
    from cortexbacktest import BrokerSimulator

    class CountingFeed(ParquetBarFeed):
        passes = 0
        def __iter__(self):
            self.passes += 1
            return super().__iter__()

    path = tmp_path / 'count.parquet'
    data = frame().head(0) if empty else frame()
    data.write_parquet(path)
    feed = CountingFeed(path, batch_rows=2)
    def run(mode):
        return BacktestEngine(broker=BrokerSimulator(initial_cash=10000),
            trade_on_close=False, build_manifest=True, data_validation=mode).run(feed, BuyOnceStrategy())
    before = run('preflight')
    assert feed.passes == 2
    after = run('stream')
    assert feed.passes == 3
    assert before.run_manifest == after.run_manifest
    assert before.summary() == after.summary()
    assert before.fills == after.fills


def test_stream_checks_adjusted_data_before_executing_that_slice(tmp_path):
    from datetime import date
    from cortexbacktest import BrokerSimulator, CorporateAction, CorporateActionType
    data = frame().with_columns(pl.lit(10.0).alias('adj_close'))
    path = tmp_path / 'adjusted.parquet'
    data.write_parquet(path)
    broker = BrokerSimulator(initial_cash=10000, corporate_actions=[
        CorporateAction('A', date(2024, 1, 2), CorporateActionType.CASH_DIVIDEND, amount=.1)])
    engine = BacktestEngine(broker=broker, data_validation='stream', build_manifest=True)
    with pytest.raises(ValueError, match='adj_close'):
        engine.run(ParquetBarFeed(path), BuyOnceStrategy())
    assert broker.fill_count == 0
    with pytest.raises(RuntimeError, match='single-use'):
        engine.run(ParquetBarFeed(path), BuyOnceStrategy())
