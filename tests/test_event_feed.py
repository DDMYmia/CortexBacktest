from datetime import UTC, datetime

import polars as pl
import pytest

from cortexbacktest import BacktestEngine, Bar, ColumnarBarFeed, Order, Side, Strategy


class BuyOnceStrategy(Strategy):
    def __init__(self) -> None:
        self.ordered = False

    def on_bars(self, bars: list[Bar]) -> list[Order]:
        if self.ordered:
            return []
        self.ordered = True
        return [Order(bar.symbol, bar.timestamp, Side.BUY, 1) for bar in bars]


def _bars() -> list[Bar]:
    return [
        Bar("MSFT", datetime(2024, 1, 2, tzinfo=UTC), 20, 20, 20, 20, volume=100),
        Bar("AAPL", datetime(2024, 1, 1, tzinfo=UTC), 10, 10, 10, 10, volume=100),
        Bar("MSFT", datetime(2024, 1, 1, tzinfo=UTC), 20, 20, 20, 20, volume=100),
        Bar("AAPL", datetime(2024, 1, 2, tzinfo=UTC), 11, 11, 11, 11, volume=100),
    ]


def test_columnar_bar_feed_groups_sorted_market_slices():
    feed = ColumnarBarFeed.from_bars(_bars())

    assert len(feed) == 2
    assert [[bar.symbol for bar in market_slice.bars] for market_slice in feed] == [["AAPL", "MSFT"], ["AAPL", "MSFT"]]


def test_backtest_engine_accepts_columnar_bar_feed():
    bars = _bars()
    list_result = BacktestEngine().run(bars, BuyOnceStrategy())
    feed_result = BacktestEngine().run(ColumnarBarFeed.from_bars(bars), BuyOnceStrategy())

    assert feed_result.performance.equity_curve().select(["timestamp", "equity"]).rows() == (
        list_result.performance.equity_curve().select(["timestamp", "equity"]).rows()
    )
    assert len(feed_result.fills) == len(list_result.fills)


def test_backtest_engine_snapshot_frequency_controls_equity_points():
    bars = [
        Bar("AAPL", datetime(2024, 1, 1, 9, 30, tzinfo=UTC), 10, 10, 10, 10, volume=100),
        Bar("AAPL", datetime(2024, 1, 1, 10, 30, tzinfo=UTC), 11, 11, 11, 11, volume=100),
        Bar("AAPL", datetime(2024, 1, 2, 9, 30, tzinfo=UTC), 12, 12, 12, 12, volume=100),
    ]

    every_bar = BacktestEngine(snapshot_frequency="bar").run(bars, BuyOnceStrategy())
    session = BacktestEngine(snapshot_frequency="session").run(bars, BuyOnceStrategy())
    final = BacktestEngine(snapshot_frequency="final").run(bars, BuyOnceStrategy())
    none = BacktestEngine(snapshot_frequency="none").run(bars, BuyOnceStrategy())

    assert every_bar.performance.equity_curve().height == 3
    assert session.performance.equity_curve().height == 2
    assert final.performance.equity_curve().height == 1
    assert none.performance.equity_curve().is_empty()


def test_columnar_bar_feed_accepts_polars_bars():
    frame = pl.DataFrame(
        {
            "date": [datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 1, 1, tzinfo=UTC)],
            "symbol": ["aapl", "msft"],
            "open": [10.0, 20.0],
            "high": [10.0, 20.0],
            "low": [10.0, 20.0],
            "close": [10.0, 20.0],
            "volume": [100.0, 100.0],
        }
    )

    feed = ColumnarBarFeed.from_polars(frame)

    assert len(feed) == 1
    assert [bar.symbol for bar in next(iter(feed)).bars] == ["AAPL", "MSFT"]


@pytest.mark.parametrize(
    "bar, message",
    [
        (Bar("AAPL", datetime(2024, 1, 1, tzinfo=UTC), 10, 9, 8, 10), "high/low"),
        (Bar("AAPL", datetime(2024, 1, 1, tzinfo=UTC), 10, 10, 10, 10, volume=-1), "volume"),
        (Bar("AAPL", datetime(2024, 1, 1, tzinfo=UTC), 10, 10, 10, 10, vwap=11), "VWAP"),
    ],
)
def test_columnar_bar_feed_rejects_impossible_ohlcv(bar, message):
    with pytest.raises(ValueError, match=message):
        ColumnarBarFeed.from_bars([bar])


def _multi_day_multi_symbol_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "date": [
                datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 1, 1, tzinfo=UTC),
                datetime(2024, 1, 2, tzinfo=UTC), datetime(2024, 1, 2, tzinfo=UTC),
                datetime(2024, 1, 3, tzinfo=UTC),
            ],
            "symbol": ["aapl", "msft", "aapl", "msft", "aapl"],
            "open": [10.0, 20.0, 11.0, 21.0, 12.0],
            "high": [10.5, 20.5, 11.5, 21.5, 12.5],
            "low": [9.5, 19.5, 10.5, 20.5, 11.5],
            "close": [10.2, 20.2, 11.2, 21.2, 12.2],
            "volume": [100.0, None, 110.0, 120.0, None],
        }
    )


def test_uncached_columnar_bar_feed_matches_cached_bars_and_fingerprint():
    frame = _multi_day_multi_symbol_frame()
    cached = ColumnarBarFeed.from_polars(frame, cache=True)
    uncached = ColumnarBarFeed.from_polars(frame, cache=False)

    assert len(uncached) == len(cached)
    assert uncached.bars == cached.bars
    assert list(uncached) == list(cached)
    assert uncached.data_fingerprint() == cached.data_fingerprint()


def test_uncached_columnar_bar_feed_can_be_iterated_more_than_once():
    feed = ColumnarBarFeed.from_polars(_multi_day_multi_symbol_frame(), cache=False)

    first_pass = list(feed)
    second_pass = list(feed)

    assert first_pass == second_pass


def test_uncached_columnar_bar_feed_runs_through_the_real_engine():
    frame = _multi_day_multi_symbol_frame()
    cached_result = BacktestEngine().run(ColumnarBarFeed.from_polars(frame, cache=True), BuyOnceStrategy())
    uncached_result = BacktestEngine().run(ColumnarBarFeed.from_polars(frame, cache=False), BuyOnceStrategy())

    assert (
        uncached_result.performance.equity_curve().select(["timestamp", "equity"]).rows()
        == cached_result.performance.equity_curve().select(["timestamp", "equity"]).rows()
    )
    assert len(uncached_result.fills) == len(cached_result.fills)


def test_uncached_columnar_bar_feed_handles_entirely_absent_optional_columns():
    # No volume/adj_close/vwap column at all: pl.lit(None).alias(...) infers a
    # Null-typed column, whose to_numpy() is float32 nan rather than float64 —
    # the case the all-nan vs. mixed-nan test above does not cover.
    frame = pl.DataFrame(
        {
            "date": [datetime(2024, 1, 1, tzinfo=UTC)],
            "symbol": ["aapl"],
            "open": [10.0], "high": [10.5], "low": [9.5], "close": [10.2],
        }
    )
    feed = ColumnarBarFeed.from_polars(frame, cache=False)
    bar = feed.bars[0]
    assert bar.volume is None
    assert bar.adj_close is None
    assert bar.vwap is None


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"high": [9.0]}, "high/low"),
        ({"volume": [-1.0]}, "volume"),
        ({"vwap": [11.0]}, "VWAP"),
    ],
)
def test_uncached_columnar_bar_feed_rejects_impossible_ohlcv(overrides, message):
    frame = pl.DataFrame(
        {
            "date": [datetime(2024, 1, 1, tzinfo=UTC)],
            "symbol": ["AAPL"],
            "open": [10.0], "high": [10.0], "low": [10.0], "close": [10.0],
            **overrides,
        }
    )
    with pytest.raises(ValueError, match=message):
        ColumnarBarFeed.from_polars(frame, cache=False)


@pytest.mark.parametrize("cache", [True, False])
@pytest.mark.parametrize("column, message", [("volume", "volume"), ("vwap", "VWAP")])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_feed_modes_reject_nonfinite_optional_market_values(cache, column, message, value):
    frame = _multi_day_multi_symbol_frame().with_columns(pl.lit(value).alias(column))
    with pytest.raises(ValueError, match=message):
        ColumnarBarFeed.from_polars(frame, cache=cache)


@pytest.mark.parametrize("cache", [True, False])
@pytest.mark.parametrize("column", ["volume", "vwap"])
def test_feed_modes_accept_null_but_reject_nan_in_same_column(cache, column):
    frame = _multi_day_multi_symbol_frame().with_columns(
        pl.Series(column, [None, None, float("nan"), None, None], dtype=pl.Float64)
    )
    with pytest.raises(ValueError, match="volume|VWAP"):
        ColumnarBarFeed.from_polars(frame, cache=cache)
    valid = frame.with_columns(pl.col(column).fill_nan(None))
    assert all(getattr(bar, column) is None for bar in ColumnarBarFeed.from_polars(valid, cache=cache).bars)


@pytest.mark.parametrize("rows", [0, 1, 3])
def test_unique_timestamp_feed_matches_uncached_and_bar_inputs(rows):
    frame = _multi_day_multi_symbol_frame().unique(subset="date", keep="first").head(rows)
    cached = ColumnarBarFeed.from_polars(frame)
    uncached = ColumnarBarFeed.from_polars(frame, cache=False)
    from_bars = ColumnarBarFeed.from_bars(uncached.bars)
    assert cached.single_bars is not None
    assert len(cached) == rows
    assert list(cached) == list(uncached) == list(from_bars)
    assert cached.data_fingerprint() == uncached.data_fingerprint()


def test_repeated_timestamp_still_groups_different_symbols():
    frame = _multi_day_multi_symbol_frame().reverse()
    cached = ColumnarBarFeed.from_polars(frame)
    assert cached.single_bars is None
    assert list(cached) == list(ColumnarBarFeed.from_polars(frame, cache=False))


@pytest.mark.parametrize("symbols", [1, 31, 32, 33, 1000])
@pytest.mark.parametrize("nullable", ["missing", "full", "mixed"])
def test_uncached_wide_slices_preserve_all_bar_fields(symbols, nullable):
    rows = symbols * 2
    optional = {
        name: [None if nullable == "missing" or (nullable == "mixed" and i % 3 == 0)
               else value for i in range(rows)]
        for name, value in {"volume": 100.0, "adj_close": 10.5, "vwap": 10.0,
                            "open_interest": 12.0, "limit_up": 15.0, "limit_down": 5.0}.items()
    }
    frame = pl.DataFrame({
        "date": [datetime(2024, 1, day, tzinfo=UTC) for day in (1, 2) for _ in range(symbols)],
        "symbol": [f"s{i:04}" for i in reversed(range(symbols))] * 2,
        "open": [10.0] * rows, "high": [11.0] * rows,
        "low": [9.0] * rows, "close": [10.5] * rows,
        "halted": [i % 2 == 0 for i in range(rows)], **optional,
    })
    cached = ColumnarBarFeed.from_polars(frame)
    uncached = ColumnarBarFeed.from_polars(frame, cache=False)
    assert list(uncached) == list(cached)
    assert list(uncached) == list(cached)
    assert uncached.data_fingerprint() == cached.data_fingerprint()
    for group in uncached:
        for bar in group.bars:
            assert type(bar.open) is float
            assert type(bar.halted) is bool
            assert bar.volume is None or type(bar.volume) is float
