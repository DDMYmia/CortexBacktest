"""When a factor value may first be used.

A factor frame keyed on `date` carries an assumption: that the value was
knowable on that date. For a price-derived factor computed from the close, it
usually is. For news sentiment, analyst revisions, filings or anything with a
publication time, it usually is not — the value stamped 2024-03-05 may have
been published at 18:00 that evening, and a strategy that trades the 2024-03-05
open on it is reading the future.

This is the third time this repository has needed the same shape. The option
chain layer states it as `ChainTimePolicy` (event_time -> available_at ->
publication lag); the bar layer states it as session hours and completion-time
labels. Factors had nothing, so a vendor export was taken at face value.

`FactorAvailability` says the two things a provider must answer, and
`align_factor_availability` turns an observation-stamped frame into one keyed on
the session the value may first be traded in. It never moves a value earlier.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Literal

import polars as pl

from ..data.backend import normalize_date_column
from ..data.calendar import TradingCalendar

__all__ = ["FactorAvailability", "align_factor_availability"]

DecisionPoint = Literal["open", "close"]


@dataclass(frozen=True, slots=True)
class FactorAvailability:
    """How an observation timestamp maps to the session that may act on it.

    - `publication_lag` is the delay between a value existing and a consumer
      being able to read it: a vendor that publishes a nightly file has a lag
      measured in hours, an intraday feed one measured in seconds. Zero is a
      claim, not a default to accept without checking.
    - `decision_point` is when the strategy acts within a session. With the
      next-open execution (`trade_on_close=False`) that is the
      open; a strategy trading the close should say `"close"`.
    - `allow_same_session` permits a value available before the decision point
      to be used in that same session. Left False, every value waits for the
      next session, which is the conservative reading and the right default for
      a source whose publication time is not trustworthy.
    """

    publication_lag: timedelta = timedelta(0)
    decision_point: DecisionPoint = "open"
    allow_same_session: bool = False

    def __post_init__(self) -> None:
        if self.publication_lag < timedelta(0):
            raise ValueError(
                "publication_lag cannot be negative: a value is not readable before it exists"
            )
        if self.decision_point not in ("open", "close"):
            raise ValueError("decision_point must be 'open' or 'close'")


def align_factor_availability(
    frame,
    calendar: TradingCalendar,
    availability: FactorAvailability | None = None,
    *,
    event_col: str = "event_time",
    date_col: str = "date",
) -> pl.DataFrame:
    """Restamp observations onto the first session that may act on them.

    Returns the frame with `date_col` set to that session, and the original
    observation kept in `event_col` so the decision stays auditable. Rows
    with no later session — observations after the calendar ends — are dropped
    rather than clamped to the last one, because clamping would invent a
    tradable moment.

    The calendar must carry session hours when `decision_point` matters; a
    date-only calendar can only answer "which day", so same-session use is
    refused rather than guessed at.
    """
    availability = availability or FactorAvailability()
    data = frame if isinstance(frame, pl.DataFrame) else pl.DataFrame(frame)
    if event_col not in data.columns:
        raise ValueError(
            f"frame requires an {event_col!r} column stating when each value was observed"
        )
    data = normalize_date_column(data, event_col)

    available = data[event_col] + availability.publication_lag
    sessions = calendar.sessions
    if availability.allow_same_session and not calendar.has_hours:
        raise ValueError(
            "allow_same_session needs a calendar with session hours: without them the decision "
            "point inside a session is unknown. Attach them with TradingCalendar.from_exchange()."
        )

    stamps = [_first_usable_session(moment, calendar, sessions, availability) for moment in available]
    result = data.with_columns(pl.Series(date_col, stamps, dtype=pl.Datetime("us")))
    return result.drop_nulls(date_col).sort(
        [date_col, *[c for c in ("symbol",) if c in result.columns]]
    )


def _first_usable_session(
    moment, calendar: TradingCalendar, sessions: pl.Series, availability: FactorAvailability
):
    if moment is None:
        return None
    if moment.tzinfo is not None and calendar.zone is None:
        raise ValueError("timezone-aware observations require a calendar zone")
    local_moment = calendar._localize(moment) if moment.tzinfo is not None else moment
    index = int(
        sessions.search_sorted(
            local_moment.replace(hour=0, minute=0, second=0, microsecond=0), side="left"
        )
    )
    while index < len(sessions):
        session = sessions[index]
        if availability.allow_same_session:
            decision = (
                calendar.session_open(session)
                if availability.decision_point == "open"
                else calendar.session_close(session)
            )
            reference = decision.replace(tzinfo=None) if moment.tzinfo is None else decision
            if moment <= reference:
                return session
        elif session > local_moment:
            # Conservative: the value waits for a session that starts strictly
            # after it was available, so a publication time inside the session
            # can never be traded on within it.
            return session
        index += 1
    return None
