"""A factor value must not be usable before a consumer could have read it."""

from datetime import datetime, time, timedelta

import polars as pl
import pytest

from cortexbacktest.data.calendar import TradingCalendar
from cortexbacktest.datasets.availability import FactorAvailability, align_factor_availability


def calendar(*, hours: bool = True) -> TradingCalendar:
    base = TradingCalendar.from_sessions(
        ["2024-03-04", "2024-03-05", "2024-03-06", "2024-03-07"]
    )
    return base.with_hours(time(9, 30), time(16, 0), zone="America/New_York") if hours else base


def news(*stamps: datetime) -> pl.DataFrame:
    return pl.DataFrame({
        "event_time": list(stamps),
        "symbol": ["AAA"] * len(stamps),
        "factor": [1.0] * len(stamps),
    })


def test_an_evening_publication_waits_for_the_next_session():
    # Published after the 2024-03-05 close: tradable at the 2024-03-06 open.
    frame = news(datetime(2024, 3, 5, 18, 0))

    aligned = align_factor_availability(frame, calendar())

    assert aligned["date"].to_list() == [datetime(2024, 3, 6)]
    # The observation is kept, so the decision can be audited afterwards.
    assert aligned["event_time"].to_list() == [datetime(2024, 3, 5, 18, 0)]


def test_publication_lag_can_push_a_value_a_further_session_out():
    frame = news(datetime(2024, 3, 5, 18, 0))

    same_day = align_factor_availability(frame, calendar())
    delayed = align_factor_availability(
        frame, calendar(), FactorAvailability(publication_lag=timedelta(hours=12))
    )

    assert same_day["date"][0] == datetime(2024, 3, 6)
    assert delayed["date"][0] == datetime(2024, 3, 7)


def test_same_session_use_requires_hours_and_respects_the_decision_point():
    morning = news(datetime(2024, 3, 5, 8, 0))  # before the 09:30 open

    conservative = align_factor_availability(morning, calendar())
    permissive = align_factor_availability(
        morning, calendar(), FactorAvailability(allow_same_session=True)
    )

    assert conservative["date"][0] == datetime(2024, 3, 6)
    assert permissive["date"][0] == datetime(2024, 3, 5)

    with pytest.raises(ValueError, match="session hours"):
        align_factor_availability(morning, calendar(hours=False),
                                  FactorAvailability(allow_same_session=True))


def test_intraday_news_cannot_be_traded_at_that_session_open():
    # 11:00 is after the open, so even permissively it waits for the next session.
    frame = news(datetime(2024, 3, 5, 11, 0))

    aligned = align_factor_availability(frame, calendar(), FactorAvailability(allow_same_session=True))

    assert aligned["date"][0] == datetime(2024, 3, 6)


def test_a_close_decision_point_can_use_the_same_session():
    frame = news(datetime(2024, 3, 5, 11, 0))

    aligned = align_factor_availability(
        frame, calendar(),
        FactorAvailability(allow_same_session=True, decision_point="close"),
    )

    assert aligned["date"][0] == datetime(2024, 3, 5)


def test_observations_past_the_calendar_are_dropped_not_clamped():
    frame = news(datetime(2024, 3, 7, 18, 0), datetime(2024, 3, 4, 6, 0))

    aligned = align_factor_availability(frame, calendar())

    # Only the March 4 morning item survives; clamping the other to March 7
    # would invent a moment at which it could have been traded.
    assert aligned["date"].to_list() == [datetime(2024, 3, 5)]


def test_a_negative_lag_is_refused():
    with pytest.raises(ValueError, match="cannot be negative"):
        FactorAvailability(publication_lag=timedelta(hours=-1))


def test_export_files_takes_the_timestamp_at_face_value_without_a_calendar():
    from cortexbacktest.adapters import normalize_factor_export

    export = pl.DataFrame({
        "analysis_timestamp": [datetime(2024, 3, 5, 18, 0)],
        "ticker": ["aaa"],
        "score": [1.5],
    })

    normalized = normalize_factor_export(export)

    # Unchanged behaviour: an 18:00 analysis is stamped on that same day, which
    # is exactly the assumption `calendar` exists to remove.
    assert normalized["date"].to_list() == [datetime(2024, 3, 5, 18, 0)]
    assert normalized["symbol"].to_list() == ["AAA"]


def test_export_files_realigns_when_given_a_calendar():
    from cortexbacktest.adapters import normalize_factor_export

    export = pl.DataFrame({
        "analysis_timestamp": [datetime(2024, 3, 5, 18, 0)],
        "ticker": ["aaa"],
        "score": [1.5],
    })

    normalized = normalize_factor_export(export, calendar=calendar())

    assert normalized["date"].to_list() == [datetime(2024, 3, 6)]
    # Provenance survives the realignment.
    assert normalized["event_time"].to_list() == [datetime(2024, 3, 5, 18, 0)]


def test_availability_without_a_calendar_is_refused_rather_than_ignored():
    from cortexbacktest.adapters import normalize_factor_export

    export = pl.DataFrame({
        "timestamp": [datetime(2024, 3, 5)], "ticker": ["aaa"], "score": [1.0],
    })

    with pytest.raises(ValueError, match="needs a calendar"):
        normalize_factor_export(export, availability=FactorAvailability())
