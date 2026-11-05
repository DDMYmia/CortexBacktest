"""Diagnostics describe a dataset; they never reject one."""

from datetime import datetime, timedelta

import polars as pl
import pytest

from cortexbacktest.data.calendar import TradingCalendar
from cortexbacktest.datasets.diagnostics import BarDiagnostics, diagnose_bars

START = datetime(2024, 1, 1)


def frame(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(rows)


def moving(symbol: str, days: range, *, base: float = 100.0) -> list[dict]:
    return [{"date": START + timedelta(days=d), "symbol": symbol, "close": base + d} for d in days]


def test_a_healthy_frame_reports_nothing_but_its_shape():
    result = diagnose_bars(frame(moving("AAA", range(30)) + moving("BBB", range(30), base=50.0)))

    assert result.sessions == 30 and result.symbols == 2
    assert not result.missing_sessions and not result.stale_runs and not result.outliers


def test_missing_sessions_are_found_against_a_calendar_and_only_against_one():
    rows = [row for row in moving("AAA", range(20))
            if row["date"] not in (START + timedelta(days=7), START + timedelta(days=8))]
    calendar = TradingCalendar.from_sessions([START + timedelta(days=d) for d in range(20)])

    with_calendar = diagnose_bars(frame(rows), calendar=calendar)
    without = diagnose_bars(frame(rows))

    assert with_calendar.missing_sessions == ("2024-01-08", "2024-01-09")
    # No calendar means no opinion about what should have been there.
    assert without.missing_sessions == ()


def test_a_price_that_never_moves_is_reported_as_stale():
    flat = [{"date": START + timedelta(days=d), "symbol": "FLAT", "close": 10.0} for d in range(12)]

    result = diagnose_bars(frame(flat + moving("AAA", range(12))))

    assert len(result.stale_runs) == 1
    assert result.stale_runs[0].startswith("FLAT x12")
    # The threshold is a parameter, not a truth about the data.
    assert diagnose_bars(frame(flat), stale_sessions=20).stale_runs == ()


def test_extreme_moves_are_counted_with_their_direction_and_date():
    rows = moving("AAA", range(10))
    rows[5]["close"] = 1000.0

    result = diagnose_bars(frame(rows), outlier_return=0.5)

    assert len(result.outliers) == 2  # up into the spike and back down out of it
    assert "AAA +" in result.outliers[0] and "2024-01-06" in result.outliers[0]


def test_zero_churn_over_a_long_sample_is_flagged_as_a_survivorship_signal():
    survivors = frame(moving("AAA", range(90)) + moving("BBB", range(90), base=20.0))

    result = diagnose_bars(survivors)

    assert result.symbol_churn == 0.0
    assert any("survivorship" in note for note in result.notes)


def test_churn_is_measured_from_the_first_session_to_the_last():
    rows = moving("AAA", range(20)) + moving("GONE", range(10))

    result = diagnose_bars(frame(rows))

    assert result.symbol_churn == pytest.approx(0.5)
    assert not any("survivorship" in note for note in result.notes)


def test_diagnostics_are_serializable_and_readable():
    result = diagnose_bars(frame(moving("AAA", range(30))))

    assert set(result.to_dict()) == {
        "sessions", "symbols", "missing_sessions", "short_sessions",
        "stale_runs", "outliers", "symbol_churn", "notes",
    }
    assert "30 sessions x 1 symbols" in result.report()
    assert isinstance(BarDiagnostics().clean, bool)


def test_a_frame_without_the_needed_columns_is_refused_before_any_scan():
    with pytest.raises(ValueError, match="normalize the frame first"):
        diagnose_bars(pl.DataFrame({"date": [START], "symbol": ["A"]}))


def intraday(days: int, per_session: int, *, truncated: dict[int, int] | None = None) -> pl.DataFrame:
    truncated = truncated or {}
    rows = []
    for day in range(days):
        count = truncated.get(day, per_session)
        for minute in range(count):
            rows.append({
                "date": START + timedelta(days=day, hours=9, minutes=30 + minute),
                "symbol": "AAA", "close": 100.0 + day + minute * 0.01,
            })
    return pl.DataFrame(rows)


def test_a_session_that_stops_early_is_found_even_though_it_is_present():
    # The real case: the minute store's SVB week has bars, they just stop.
    # A session-count check sees nothing wrong.
    result = diagnose_bars(intraday(20, 60, truncated={11: 12, 12: 9}))

    assert result.missing_sessions == ()  # every session is present
    assert len(result.short_sessions) == 2
    assert "9/60 observations" in result.short_sessions[0]


def test_short_sessions_are_silent_on_one_row_per_session_data():
    # Aggregating minutes to daily bars erases the evidence, and the check says
    # nothing rather than inventing a finding from equal counts.
    daily = frame(moving("AAA", range(30)))

    assert diagnose_bars(daily).short_sessions == ()


def test_the_short_session_threshold_is_a_parameter():
    data = intraday(20, 60, truncated={11: 40})

    assert diagnose_bars(data).short_sessions == ()  # 40/60 is above the default half
    assert diagnose_bars(data, short_session_fraction=0.8).short_sessions != ()
