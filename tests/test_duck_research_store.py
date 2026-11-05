"""Tests for the DuckDB query layer over the Parquet research layer.

Written against real Parquet files in a tmp directory, laid out exactly like the
research layer (`year=YYYY/part-YYYY-MM.parquet`), because the behaviour worth
testing here is what DuckDB does with hive partitions and timezones -- neither
of which a mock would reproduce.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from cortexbacktest.datasets import ResearchStore

pytest.importorskip("duckdb")


def _write_layer(root, rows: list[dict]) -> None:
    """Write rows into year= partitions, as build_research_layer does."""
    frame = pl.DataFrame(rows)
    for (year,), part in frame.group_by(["year"]):
        directory = root / f"year={year}"
        directory.mkdir(parents=True, exist_ok=True)
        part.drop("year").write_parquet(directory / f"part-{year}-01.parquet")


def _bar(ticker: str, when: datetime, close: float = 100.0, volume: float = 1_000.0) -> dict:
    return {
        "timestamp": when,
        "open": close - 1,
        "high": close + 2,
        "low": close - 2,
        "close": close,
        "volume": volume,
        "ticker": ticker,
        "year": when.year,
    }


@pytest.fixture
def store(tmp_path):
    """One US session plus its extended hours, which straddles a UTC midnight.

    2023-03-01 in New York is UTC-5. The 09:30 open is 14:30 UTC and the 19:00
    after-hours bar is 00:00 UTC on 2023-03-02 -- so this session spans two UTC
    dates and one exchange date, which is the whole point.
    """
    session = datetime(2023, 3, 1, 14, 30, tzinfo=UTC)  # 09:30 New York
    rows = [
        _bar("AAPL", session, close=100.0, volume=500.0),
        _bar("AAPL", session + timedelta(hours=1), close=102.0, volume=700.0),
        _bar("AAPL", session + timedelta(hours=6, minutes=29), close=101.0, volume=300.0),
        # 19:00 New York -- 00:00 UTC the next day.
        _bar("AAPL", datetime(2023, 3, 2, 0, 0, tzinfo=UTC), close=99.0, volume=100.0),
        # A second, ordinary session.
        _bar("AAPL", datetime(2023, 3, 2, 14, 30, tzinfo=UTC), close=105.0, volume=900.0),
        _bar("MSFT", session, close=250.0, volume=400.0),
        _bar("MSFT", datetime(2022, 6, 1, 14, 30, tzinfo=UTC), close=240.0, volume=100.0),
    ]
    _write_layer(tmp_path, rows)
    return ResearchStore(root=str(tmp_path), threads=2, memory_limit="512MB")


class TestSessionBucketing:
    def test_a_session_and_its_after_hours_are_one_day(self, store) -> None:
        # The bug this module exists to prevent: grouping on the UTC date puts
        # the 19:00 New York bar on the following day and invents a session.
        daily = store.daily_bars("AAPL")
        assert daily["session_date"].to_list() == [
            __import__("datetime").date(2023, 3, 1),
            __import__("datetime").date(2023, 3, 2),
        ]

    def test_utc_bucketing_assigns_the_after_hours_bar_to_the_wrong_day(self, store) -> None:
        # Same two dates either way here, but not the same sessions: by UTC the
        # 19:00 New York bar is filed under 2023-03-02 and the first session
        # loses it.
        by_utc = dict(
            store.sql(
                "SELECT timestamp::DATE, count(*) FROM bars WHERE ticker='AAPL' GROUP BY 1"
            )
        )
        by_session = dict(
            zip(
                store.daily_bars("AAPL")["session_date"].to_list(),
                store.daily_bars("AAPL")["bar_count"].to_list(),
                strict=True,
            )
        )
        assert list(by_utc.values()) != list(by_session.values())
        assert sorted(by_utc.values()) == [2, 3]
        assert sorted(by_session.values()) == [1, 4]

    def test_a_lone_after_hours_bar_invents_a_session_under_utc(self, tmp_path) -> None:
        # The damaging shape, isolated: one session whose last bar falls after
        # UTC midnight. By UTC that is two trading days; there was one. Over a
        # year this is how AAPL 2023 becomes 303 sessions instead of 250.
        rows = [
            _bar("Z", datetime(2023, 3, 1, 14, 30, tzinfo=UTC)),
            _bar("Z", datetime(2023, 3, 2, 0, 30, tzinfo=UTC)),  # 19:30 New York
        ]
        _write_layer(tmp_path, rows)
        store = ResearchStore(root=str(tmp_path))
        assert store.sql("SELECT count(DISTINCT timestamp::DATE) FROM bars")[0][0] == 2
        assert len(store.daily_bars("Z")) == 1

    def test_open_and_close_are_first_and_last_by_time(self, store) -> None:
        # Not min and max, which would report the day's cheapest and dearest
        # prices as its open and close.
        first = store.daily_bars("AAPL").row(0, named=True)
        assert first["open"] == 99.0  # 100.0 - 1, the 09:30 bar
        assert first["close"] == 99.0  # the 19:00 after-hours bar
        assert first["high"] == 104.0
        assert first["low"] == 97.0

    def test_the_after_hours_volume_is_included(self, store) -> None:
        assert store.daily_bars("AAPL").row(0, named=True)["volume"] == 1600.0

    def test_bar_count_exposes_a_thin_session(self, store) -> None:
        counts = store.daily_bars("AAPL")["bar_count"].to_list()
        assert counts == [4, 1]


class TestFiltering:
    def test_a_date_window_is_applied_on_the_session_date(self, store) -> None:
        assert len(store.daily_bars("AAPL", start="2023-03-02")) == 1
        assert len(store.daily_bars("AAPL", end="2023-03-01")) == 1

    def test_an_unknown_symbol_returns_no_rows_rather_than_raising(self, store) -> None:
        assert len(store.daily_bars("NOPE")) == 0

    def test_the_hive_year_column_is_queryable(self, store) -> None:
        years = {row[0] for row in store.sql("SELECT DISTINCT year FROM bars")}
        assert years == {2022, 2023}


class TestSurveys:
    def test_universe_by_year_counts_distinct_tickers(self, store) -> None:
        frame = store.universe_by_year()
        assert frame.to_dicts() == [
            {"year": 2022, "symbols": 1},
            {"year": 2023, "symbols": 2},
        ]

    def test_liquidity_ranking_orders_by_traded_value(self, store) -> None:
        frame = store.liquidity_ranking("2023-03-01", "2023-03-02")
        # AAPL trades five bars over the window against MSFT's one, so it leads
        # on value despite the lower price.
        assert frame["ticker"].to_list() == ["AAPL", "MSFT"]
        assert frame["dollar_volume"].to_list() == [256_100.0, 100_000.0]

    def test_liquidity_ranking_reports_coverage_beside_the_value(self, store) -> None:
        # A name can rank on a few frantic days and be untradeable otherwise.
        frame = store.liquidity_ranking("2023-03-01", "2023-03-02")
        rows = {row["ticker"]: row for row in frame.to_dicts()}
        assert rows["AAPL"]["sessions"] == 2
        assert rows["MSFT"]["sessions"] == 1

    def test_session_coverage_puts_the_thinnest_session_first(self, store) -> None:
        frame = store.session_coverage("AAPL", 2023)
        assert frame["bars"].to_list() == [1, 4]


class TestEscapeHatch:
    def test_arbitrary_sql_runs_against_the_registered_view(self, store) -> None:
        assert store.sql("SELECT count(*) FROM bars")[0][0] == 7

    def test_parameters_are_bound_not_interpolated(self, store) -> None:
        rows = store.sql("SELECT count(*) FROM bars WHERE ticker = ?", ["AAPL"])
        assert rows[0][0] == 5

    def test_frame_returns_polars(self, store) -> None:
        assert isinstance(store.frame("SELECT * FROM bars"), pl.DataFrame)

    def test_the_configured_timezone_is_the_one_used(self, tmp_path) -> None:
        # Same bars, a different exchange: the session boundary moves with it.
        _write_layer(
            tmp_path, [_bar("X", datetime(2023, 3, 2, 0, 0, tzinfo=UTC))]
        )
        ny = ResearchStore(root=str(tmp_path), exchange_tz="America/New_York")
        tokyo = ResearchStore(root=str(tmp_path), exchange_tz="Asia/Tokyo")
        assert ny.daily_bars("X")["session_date"][0].isoformat() == "2023-03-01"
        assert tokyo.daily_bars("X")["session_date"][0].isoformat() == "2023-03-02"
