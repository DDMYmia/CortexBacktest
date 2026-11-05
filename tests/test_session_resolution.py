"""Session hours, and the resolution contract that separates the two bar lines."""

from datetime import UTC, date, datetime, time, timedelta

import polars as pl
import pytest

from cortexbacktest.contracts import Bar
from cortexbacktest.data.calendar import (
    SessionWindow,
    TradingCalendar,
    infer_session_hours,
)
from cortexbacktest.datasets.loaders import load_backtest_dataset
from cortexbacktest.datasets.schema import bars_to_events, normalize_bars, validate_bars
from cortexbacktest.event.timeframes import bucket_end, resample_bars

HALF_DAY = date(2024, 11, 29)


def calendar() -> TradingCalendar:
    return TradingCalendar.business_days("2024-11-25", "2024-11-29").with_hours(
        time(9, 30), time(16, 0), zone="America/New_York",
        overrides={HALF_DAY: SessionWindow(time(9, 30), time(13, 0))},
    )


def test_calendar_without_hours_refuses_instant_questions():
    plain = TradingCalendar.business_days("2024-11-25", "2024-11-29")

    assert not plain.has_hours
    assert plain.contains("2024-11-28")
    with pytest.raises(ValueError, match="no session hours"):
        plain.is_open(datetime(2024, 11, 28, 10, 0))


def test_session_hours_track_early_closes_and_reject_bad_windows():
    cal = calendar()

    assert cal.session_close(HALF_DAY) == datetime(2024, 11, 29, 13, 0, tzinfo=cal.session_close(HALF_DAY).tzinfo)
    assert cal.session_close("2024-11-28").hour == 16
    assert cal.is_open(datetime(2024, 11, 29, 12, 59))
    # The close is exclusive, and pre-market is outside the session.
    assert not cal.is_open(datetime(2024, 11, 29, 13, 0))
    assert not cal.is_open(datetime(2024, 11, 28, 9, 0))
    with pytest.raises(ValueError, match="must precede close"):
        SessionWindow(time(16, 0), time(9, 30))
    with pytest.raises(ValueError, match="require a zone"):
        TradingCalendar(cal.sessions, hours=SessionWindow(time(9, 30), time(16, 0)))


def test_aware_and_naive_timestamps_agree_on_the_session():
    cal = calendar()
    naive = datetime(2024, 11, 28, 15, 59)
    aware = datetime(2024, 11, 28, 20, 59, tzinfo=UTC)  # the same instant in EST

    assert cal.session_of(naive) == cal.session_of(aware) == datetime(2024, 11, 28)


def test_buckets_end_at_the_close_not_past_it():
    cal = calendar()

    # Without a calendar, wall-clock arithmetic runs past a close nothing traded at.
    assert bucket_end(datetime(2024, 11, 28, 15, 0), "3h") == datetime(2024, 11, 28, 18, 0)
    assert bucket_end(datetime(2024, 11, 28, 15, 0), "3h", calendar=cal) == datetime(2024, 11, 28, 16, 0)
    assert bucket_end(datetime(2024, 11, 29, 12, 0), "2h", calendar=cal) == datetime(2024, 11, 29, 13, 0)
    # A bucket that closes inside the session is untouched.
    assert bucket_end(datetime(2024, 11, 28, 10, 0), "1h", calendar=cal) == datetime(2024, 11, 28, 11, 0)


def test_resampled_final_bucket_is_available_at_the_early_close():
    cal = calendar()
    bars = [Bar("X", datetime(2024, 11, 29, 12, 45 + minute), 10.0, 10.0, 10.0, 10.0) for minute in range(15)]

    stamps = [bar.timestamp for bar in resample_bars(bars, "2h", calendar=cal)]

    assert stamps == [datetime(2024, 11, 29, 13, 0)]


def test_inferred_hours_read_the_half_day_out_of_the_data():
    stamps = []
    for day, close_hour in ((27, 16), (28, 16), (29, 13)):
        moment = datetime(2024, 11, day, 9, 30)
        while moment < datetime(2024, 11, day, close_hour, 0):
            stamps.append(moment)
            moment += timedelta(minutes=30)

    inferred = infer_session_hours(stamps, zone="America/New_York", bar_length=timedelta(minutes=30))

    assert inferred.hours == SessionWindow(time(9, 30), time(16, 0))
    assert inferred.session_overrides == {HALF_DAY: SessionWindow(time(9, 30), time(13, 0))}
    with pytest.raises(ValueError, match="single observation"):
        infer_session_hours([datetime(2024, 11, 27, 9, 30)], zone="America/New_York")


def intraday_frame() -> pl.DataFrame:
    rows = [
        {"date": datetime(2024, 11, 28, 9, 30 + minute), "symbol": "X",
         "open": 10.0, "high": 10.0, "low": 10.0, "close": 10.0, "volume": 1.0, "adj_close": 10.0}
        for minute in range(3)
    ]
    return normalize_bars(pl.DataFrame(rows))


def test_session_resolution_rejects_intraday_bars():
    report = validate_bars(intraday_frame(), resolution="session")

    assert not report.ok
    assert "more than one observation" in report.errors[0]
    assert "X@2024-11-28x3" in report.errors[0]
    # The same frame is fine where every row is meant to be an instant.
    assert validate_bars(intraday_frame(), resolution="any").ok
    assert validate_bars(intraday_frame(), resolution="intraday", calendar=calendar()).ok


def test_intraday_resolution_rejects_bars_outside_session_hours():
    frame = normalize_bars(pl.DataFrame([
        {"date": datetime(2024, 11, 29, 14, 0), "symbol": "X",
         "open": 10.0, "high": 10.0, "low": 10.0, "close": 10.0, "volume": 1.0, "adj_close": 10.0},
    ]))

    report = validate_bars(frame, resolution="intraday", calendar=calendar())

    assert not report.ok
    assert "outside session hours" in report.errors[0]


def test_factor_research_entry_point_refuses_intraday_data():
    bars = intraday_frame()
    factors = pl.DataFrame({"date": [datetime(2024, 11, 28)], "symbol": ["X"], "factor": [1.0]})

    with pytest.raises(ValueError, match="Intraday data does not belong"):
        load_backtest_dataset(bars=bars, factors=factors)


def test_event_path_still_accepts_the_same_intraday_bars():
    bars = bars_to_events(intraday_frame())

    assert [bar.timestamp.minute for bar in bars] == [30, 31, 32]
    with pytest.raises(ValueError, match="outside session hours"):
        bars_to_events(intraday_frame(), resolution="intraday", calendar=calendar().with_hours(
            time(10, 0), time(16, 0), zone="America/New_York"))


def intraday_matrix() -> pl.DataFrame:
    """Two sessions of three minute closes, with a gap over the night."""
    return pl.DataFrame({
        "date": [
            datetime(2024, 11, 27, 15, 57), datetime(2024, 11, 27, 15, 58), datetime(2024, 11, 27, 15, 59),
            datetime(2024, 11, 28, 9, 30), datetime(2024, 11, 28, 9, 31), datetime(2024, 11, 28, 9, 32),
        ],
        "X": [100.0, 101.0, 102.0, 120.0, 121.0, 122.0],
    })


def test_adjacent_row_returns_refuse_intraday_data_without_a_calendar():
    from cortexbacktest.research.frames import returns_matrix

    with pytest.raises(ValueError, match="without a calendar"):
        returns_matrix(intraday_matrix())


def test_intraday_returns_stop_at_the_session_boundary():
    from cortexbacktest.research.frames import overnight_returns_matrix, returns_matrix

    cal = TradingCalendar.business_days("2024-11-25", "2024-11-29").with_hours(
        time(9, 30), time(16, 0), zone="America/New_York")
    intraday = returns_matrix(intraday_matrix(), calendar=cal)["X"].to_list()
    overnight = overnight_returns_matrix(intraday_matrix(), calendar=cal)["X"].to_list()

    # The 102 -> 120 step is an overnight gap, not an intraday return.
    assert intraday[3] is None
    assert intraday[4] == pytest.approx(1 / 120)
    assert overnight[3] == pytest.approx(18 / 102)
    # Every adjacent-row move lands in exactly one of the two series.
    assert all((a is None) != (b is None) for a, b in list(zip(intraday, overnight))[1:])


def test_forward_returns_do_not_reach_into_the_next_session():
    from cortexbacktest.research.frames import forward_returns_matrix

    cal = TradingCalendar.business_days("2024-11-25", "2024-11-29").with_hours(
        time(9, 30), time(16, 0), zone="America/New_York")

    forward = forward_returns_matrix(intraday_matrix(), periods=2, calendar=cal)["X"].to_list()

    # From 15:57 the two-bar horizon still ends inside the session; from 15:58 it does not.
    assert forward[0] == pytest.approx(2 / 100)
    assert forward[1] is None and forward[2] is None


def test_daily_matrices_are_untouched_by_the_session_logic():
    from cortexbacktest.research.frames import forward_returns_matrix, returns_matrix

    daily = pl.DataFrame({
        "date": [datetime(2024, 11, 25), datetime(2024, 11, 26), datetime(2024, 11, 27)],
        "X": [100.0, 110.0, 121.0],
    })

    assert returns_matrix(daily)["X"].to_list() == [None, pytest.approx(0.1), pytest.approx(0.1)]
    assert forward_returns_matrix(daily, periods=1)["X"][0] == pytest.approx(0.1)


def test_exchange_calendar_supplies_the_early_closes_the_data_cannot():
    pytest.importorskip("pandas_market_calendars")

    calendar = TradingCalendar.from_exchange("XNYS", "2023-01-01", "2023-12-31")

    assert calendar.zone == "America/New_York"
    assert calendar.hours == SessionWindow(time(9, 30), time(16, 0))
    # 2023's only 1:00pm closes: July 3 and the day after Thanksgiving.
    assert sorted(calendar.session_overrides) == [date(2023, 7, 3), date(2023, 11, 24)]
    assert calendar.session_close("2023-11-24").hour == 13
    assert calendar.session_close("2023-11-22").hour == 16
    assert not calendar.contains("2023-11-23")  # Thanksgiving itself is closed


def test_exchange_hours_and_inferred_hours_are_interchangeable_at_the_call_site():
    pytest.importorskip("pandas_market_calendars")
    from cortexbacktest.event.timeframes import bucket_end

    exchange = TradingCalendar.from_exchange("XNYS", "2023-11-01", "2023-11-30")
    inferred = infer_session_hours(
        [datetime(2023, 11, day, hour, minute)
         for day in (21, 22, 24) for hour in range(9, 16) for minute in (30, 45)],
        zone="America/New_York",
    )

    # Same regular window from both, but only the exchange knows about the half day.
    assert exchange.hours.open == inferred.hours.open == time(9, 30)
    assert date(2023, 11, 24) in exchange.session_overrides
    assert bucket_end(datetime(2023, 11, 24, 12, 0), "2h", calendar=exchange) == datetime(2023, 11, 24, 13, 0)


def intraday_dataset_frames():
    """Two symbols, two sessions, three minute bars each."""
    rows = []
    for day in (27, 28):
        for minute in range(3):
            for symbol in ("X", "Y"):
                rows.append({
                    "date": datetime(2024, 11, day, 9, 30 + minute), "symbol": symbol,
                    "open": 10.0, "high": 10.5, "low": 9.5, "close": 10.0 + minute,
                    "volume": 100.0, "adj_close": 10.0 + minute,
                })
    bars = normalize_bars(pl.DataFrame(rows))
    factors = pl.DataFrame({
        "date": [row["date"] for row in rows],
        "symbol": [row["symbol"] for row in rows],
        "factor": [float(index) for index in range(len(rows))],
    })
    return bars, factors


def test_intraday_research_needs_a_calendar_that_knows_the_sessions():
    bars, factors = intraday_dataset_frames()
    plain = TradingCalendar.from_sessions(["2024-11-27", "2024-11-28"])

    with pytest.raises(ValueError, match="session hours"):
        load_backtest_dataset(bars=bars, factors=factors, calendar=plain, resolution="intraday")


def test_intraday_research_runs_and_drops_the_overnight_step():
    bars, factors = intraday_dataset_frames()
    calendar = TradingCalendar.from_sessions(["2024-11-27", "2024-11-28"]).with_hours(
        time(9, 30), time(16, 0), zone="America/New_York")

    dataset = load_backtest_dataset(
        bars=bars, factors=factors, calendar=calendar, resolution="intraday")

    returns = dataset.returns["X"].to_list()
    # Six rows: first of each session is null (no predecessor inside it), the
    # other four are genuine intraday steps.
    assert len(returns) == 6
    assert returns[0] is None and returns[3] is None
    assert all(value is not None for index, value in enumerate(returns) if index not in (0, 3))


def test_session_resolution_is_still_the_default_and_still_refuses_intraday():
    bars, factors = intraday_dataset_frames()

    with pytest.raises(ValueError, match="Intraday data does not belong"):
        load_backtest_dataset(bars=bars, factors=factors)
