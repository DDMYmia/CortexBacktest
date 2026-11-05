from datetime import datetime

import polars as pl
import pytest

from cortexbacktest.data import TradingCalendar, Universe, apply_universe_mask


def test_trading_calendar_business_days_and_neighbors():
    calendar = TradingCalendar.business_days("2024-01-01", "2024-01-05")

    assert calendar.contains("2024-01-02")
    assert not calendar.contains("2024-01-06")
    assert calendar.next_session("2024-01-02") == datetime(2024, 1, 3)
    assert calendar.previous_session("2024-01-03") == datetime(2024, 1, 2)


def test_trading_calendar_neighbors_skip_weekends_and_edges():
    calendar = TradingCalendar.business_days("2024-01-01", "2024-01-08")

    # Friday -> Monday, skipping the weekend.
    assert calendar.next_session("2024-01-05") == datetime(2024, 1, 8)
    assert calendar.previous_session("2024-01-08") == datetime(2024, 1, 5)
    assert calendar.next_session("2024-01-08") is None
    assert calendar.previous_session("2024-01-01") is None


def test_trading_calendar_align_reindexes_onto_sessions():
    calendar = TradingCalendar.business_days("2024-01-01", "2024-01-05")
    frame = pl.DataFrame(
        {"date": ["2024-01-01", "2024-01-02", "2024-01-06"], "close": [10.0, 11.0, 12.0]}
    )

    aligned = calendar.align(frame)

    assert aligned["date"].to_list() == calendar.sessions.to_list()
    row = aligned.filter(pl.col("date") == datetime(2024, 1, 2)).row(0, named=True)
    assert row["close"] == 11.0
    # Missing session is null; the off-calendar Saturday row is dropped.
    assert aligned.filter(pl.col("date") == datetime(2024, 1, 3)).row(0, named=True)["close"] is None
    assert aligned.filter(pl.col("date") == datetime(2024, 1, 6)).height == 0


def test_trading_calendar_rejects_off_calendar_timestamps():
    calendar = TradingCalendar.business_days("2024-01-01", "2024-01-05")
    with pytest.raises(ValueError, match="outside calendar"):
        calendar.validate_timestamps(["2024-01-06"], label="bars")


def test_universe_from_static_symbols_masks_frame():
    frame = pl.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02"],
            "AAPL": [1.0, 2.0],
            "MSFT": [3.0, 4.0],
            "TSLA": [5.0, 6.0],
        }
    )
    universe = Universe.from_symbols(["2024-01-01", "2024-01-02"], ["aapl", "msft"])

    masked = universe.apply(frame)

    assert masked["AAPL"].to_list() == [1.0, 2.0]
    assert masked["MSFT"].to_list() == [3.0, 4.0]
    # TSLA is out of universe on every date.
    assert masked["TSLA"].to_list() == [None, None]


def test_universe_from_membership_is_point_in_time():
    membership = pl.DataFrame(
        {
            "symbol": ["AAPL", "MSFT"],
            "start": ["2024-01-01", "2024-01-03"],
            "end": ["2024-01-02", None],
        }
    )
    dates = ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]
    universe = Universe.from_membership(membership, dates)
    frame = pl.DataFrame({"date": dates, "AAPL": [1.0, 2.0, 3.0, 4.0], "MSFT": [5.0, 6.0, 7.0, 8.0]})

    masked = apply_universe_mask(frame, universe)

    def cell(symbol: str, day: int) -> float | None:
        return masked.filter(pl.col("date") == datetime(2024, 1, day)).row(0, named=True)[symbol]

    # AAPL only 01-01..01-02; MSFT only from 01-03 (open-ended end).
    assert cell("AAPL", 1) == 1.0
    assert cell("AAPL", 3) is None
    assert cell("MSFT", 2) is None
    assert cell("MSFT", 3) == 7.0


def test_universe_rejects_empty_inputs():
    with pytest.raises(ValueError, match="dates must not be empty"):
        Universe.from_symbols([], ["AAPL"])
    with pytest.raises(ValueError, match="symbols must not be empty"):
        Universe.from_symbols(["2024-01-01"], [])
