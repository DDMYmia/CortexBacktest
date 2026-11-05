"""Trading calendar primitives.

Sessions are a sorted, unique ``Datetime`` Polars Series -- there is no index to
carry, so neighbour lookups (``next_session`` / ``previous_session``) are binary
searches over that Series rather than boolean-mask scans, and ``align`` is an
explicit left join on a date column.

A calendar answers two different questions, and which one it can answer depends
on whether session hours were declared. Without hours it knows only *which days
are sessions* -- enough for daily bars, where the timestamp is a session label
rather than an instant. With hours (``with_hours`` / ``infer_session_hours``) it
also knows *which instants are inside a session*, which is what intraday bars,
bucket boundaries at the close, and availability timestamps need.

Naive timestamps are read as wall clock in the calendar zone. That is the same
convention the timeframe bucketing already uses, and it is stated here so the
two layers cannot drift apart.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import polars as pl

from .backend import normalize_date_column


def _to_datetime(value: str | date | datetime) -> datetime:
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day)
    parsed = pl.Series([value]).str.to_datetime(strict=False)[0]
    if parsed is None:
        raise ValueError(f"could not parse date: {value!r}")
    return datetime(parsed.year, parsed.month, parsed.day)


@dataclass(frozen=True, slots=True)
class SessionWindow:
    """Wall-clock open/close of one session, in the owning calendar's zone.

    ``close`` is exclusive: a bar stamped exactly at the close belongs to the
    next bucket, not to this session. Half days are ordinary windows with an
    earlier close, carried in ``TradingCalendar.session_overrides``.
    """

    open: time
    close: time

    def __post_init__(self) -> None:
        if self.open >= self.close:
            raise ValueError(f"session open {self.open} must precede close {self.close}")


@dataclass(frozen=True, slots=True)
class TradingCalendar:
    """A simple session calendar.

    Exchange-specific calendars come in through :meth:`from_exchange` and the
    optional ``calendar`` extra.

    ``hours`` and ``zone`` are optional together: a calendar without them is
    date-only and every instant-level query raises rather than guessing a
    close. Attach them with :meth:`with_hours`, take them from the exchange
    with :meth:`from_exchange`, or read what a data feed actually spans with
    :func:`infer_session_hours`.

    Those last two are not interchangeable, and which one is right depends on
    the question. The exchange knows when trading was allowed; the feed knows
    when it has bars. An early close is only visible to the first -- post-market
    prints continue past it and are indistinguishable in OHLCV -- while a
    vendor's missing morning is only visible to the second.
    """

    sessions: pl.Series
    name: str = "custom"
    zone: str | None = None
    hours: SessionWindow | None = None
    session_overrides: Mapping[date, SessionWindow] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.hours is not None and self.zone is None:
            raise ValueError("session hours require a zone; wall clock without a zone is not an instant")
        if self.hours is None and self.session_overrides:
            raise ValueError("session overrides require regular hours to override")

    @property
    def has_hours(self) -> bool:
        """Whether this calendar can answer instant-level questions."""
        return self.hours is not None

    def with_hours(
        self,
        open: time,
        close: time,
        *,
        zone: str,
        overrides: Mapping[date, SessionWindow] | None = None,
    ) -> "TradingCalendar":
        """Return a copy that also knows when each session opens and closes."""
        return replace(
            self,
            zone=zone,
            hours=SessionWindow(open, close),
            session_overrides=dict(overrides or {}),
        )

    def window(self, day: str | date | datetime) -> SessionWindow:
        """Wall-clock window for a session, honouring half-day overrides."""
        session = _to_datetime(day)
        if not self.contains(session):
            raise ValueError(f"{session.date().isoformat()} is not a session on calendar {self.name}")
        return self.session_window(session)

    def session_window(self, day: str | date | datetime) -> SessionWindow:
        """Wall-clock window for a day, without checking it is a session."""
        self._require_hours()
        assert self.hours is not None
        return self.session_overrides.get(_to_datetime(day).date(), self.hours)

    def session_open(self, day: str | date | datetime) -> datetime:
        """Zone-aware instant this session opens."""
        return self._instant(_to_datetime(day).date(), self.window(day).open)

    def session_close(self, day: str | date | datetime) -> datetime:
        """Zone-aware instant this session closes (exclusive)."""
        return self._instant(_to_datetime(day).date(), self.window(day).close)

    def is_open(self, timestamp: datetime) -> bool:
        """Whether an instant falls inside a session window, close exclusive."""
        return self.session_of(timestamp) is not None

    def session_of(self, timestamp: datetime) -> datetime | None:
        """Midnight-keyed session an instant belongs to, or None if outside one.

        Sessions do not span midnight here, so an instant maps to at most the
        session of its own local calendar day. Overnight markets need a real
        session index and should not be squeezed into this shape.
        """
        self._require_hours()
        local = self._localize(timestamp)
        session = datetime(local.year, local.month, local.day)
        if not self.contains(session):
            return None
        window = self.session_window(session)
        return session if window.open <= local.time() < window.close else None

    def clamp_to_close(self, timestamp: datetime, candidate: datetime) -> datetime:
        """Pull a bucket end back to the session close of ``timestamp``.

        A 30-minute bucket opened at 15:45 ET ends at 16:15 by wall-clock
        arithmetic, an instant at which nothing trades. Clamping keeps the
        bucket's availability timestamp on a moment the market actually
        reached. Instants outside a session are returned untouched: inventing a
        close for them would be the same guess this class refuses to make.
        """
        session = self.session_of(timestamp)
        if session is None:
            return candidate
        # Compare in the zone's wall clock: that is the one frame both a naive
        # and an aware candidate can be read in without assuming a UTC offset.
        if self._localize(candidate) <= datetime.combine(session.date(), self.session_window(session).close):
            return candidate
        close = self.session_close(session)
        return close.astimezone(candidate.tzinfo) if candidate.tzinfo else close.replace(tzinfo=None)

    def _require_hours(self) -> None:
        if self.hours is None:
            raise ValueError(
                f"calendar {self.name} has no session hours; "
                "attach them with with_hours() or infer_session_hours()"
            )

    def _instant(self, day: date, moment: time) -> datetime:
        assert self.zone is not None
        return datetime.combine(day, moment, tzinfo=ZoneInfo(self.zone))

    def _localize(self, timestamp: datetime) -> datetime:
        assert self.zone is not None
        if timestamp.tzinfo is None:
            # Naive input is wall clock in the calendar zone; see module docstring.
            return timestamp
        return timestamp.astimezone(ZoneInfo(self.zone)).replace(tzinfo=None)

    @classmethod
    def business_days(
        cls,
        start: str | date,
        end: str | date,
        *,
        name: str = "business_days",
    ) -> "TradingCalendar":
        days = pl.date_range(_to_datetime(start).date(), _to_datetime(end).date(), interval="1d", eager=True)
        business = days.filter(days.dt.weekday() <= 5).cast(pl.Datetime("us"))
        return cls(business.rename("session"), name=name)

    @classmethod
    def from_sessions(cls, sessions, *, name: str = "custom") -> "TradingCalendar":
        return cls(_normalize_sessions(sessions), name=name)

    @classmethod
    def from_exchange(
        cls,
        code: str,
        start: str | date,
        end: str | date,
        *,
        name: str | None = None,
    ) -> "TradingCalendar":
        """Sessions and hours as the exchange published them.

        This is where early closes come from. They cannot be recovered from a
        price feed -- trading continues past a 1:00pm close and the prints look
        the same -- so a study that buckets or settles at the close needs the
        exchange's own schedule, not the data's.

        The result has the same shape :func:`infer_session_hours` produces: a
        modal regular window plus one override per session that differs. The
        two are interchangeable at the call site and differ only in what they
        are evidence of.

        Requires the optional ``calendar`` extra.
        """
        try:
            import pandas_market_calendars as market_calendars
        except ImportError as exc:  # pragma: no cover - exercised by the extra's absence
            raise ImportError(
                "Install cortexbacktest[calendar] to build calendars from exchange schedules"
            ) from exc

        exchange = market_calendars.get_calendar(code)
        schedule = exchange.schedule(
            start_date=_to_datetime(start).date().isoformat(),
            end_date=_to_datetime(end).date().isoformat(),
        )
        if schedule.empty:
            raise ValueError(f"exchange {code} has no sessions between {start} and {end}")
        zone = str(exchange.tz)
        # The schedule carries UTC instants; the window is wall clock, so the
        # conversion has to happen before the time-of-day is read or every
        # daylight-saving change shifts the session by an hour.
        local = schedule[["market_open", "market_close"]].apply(lambda column: column.dt.tz_convert(zone))
        windows = {
            stamp.date(): SessionWindow(opened.time(), closed.time())
            for stamp, opened, closed in local.itertuples(name=None)
        }
        counts = Counter(windows.values())
        regular = max(counts, key=lambda window: (counts[window], window.close))
        overrides = {day: window for day, window in windows.items() if window != regular}
        calendar = cls.from_sessions(sorted(windows), name=name or code)
        return calendar.with_hours(regular.open, regular.close, zone=zone, overrides=overrides)

    def contains(self, timestamp: str | date) -> bool:
        target = _to_datetime(timestamp)
        index = self.sessions.search_sorted(target, side="left")
        return index < len(self.sessions) and self.sessions[index] == target

    def between(self, start: str | date, end: str | date) -> pl.Series:
        start_ts = _to_datetime(start)
        end_ts = _to_datetime(end)
        return self.sessions.filter((self.sessions >= start_ts) & (self.sessions <= end_ts))

    def next_session(self, timestamp: str | date) -> datetime | None:
        target = _to_datetime(timestamp)
        # First session strictly after target: the right insertion point.
        index = self.sessions.search_sorted(target, side="right")
        return self.sessions[index] if index < len(self.sessions) else None

    def previous_session(self, timestamp: str | date) -> datetime | None:
        target = _to_datetime(timestamp)
        # Last session strictly before target: one left of the left insertion point.
        index = self.sessions.search_sorted(target, side="left")
        return self.sessions[index - 1] if index > 0 else None

    def align(self, frame: pl.DataFrame, *, on: str = "date") -> pl.DataFrame:
        """Reindex a date-keyed frame onto this calendar's sessions.

        Rows outside the calendar are dropped and missing sessions are filled
        with nulls -- the same contract a reindex gives, but with the join key
        written out explicitly.
        """
        normalized = normalize_date_column(frame, on)
        spine = self.sessions.alias(on).to_frame()
        return spine.join(normalized, on=on, how="left").sort(on)

    def validate_timestamps(self, timestamps, *, label: str = "timestamps", instants: bool = False) -> None:
        """Reject observations that fall outside the declared session calendar.

        By default only the *date* is checked, which is all a session-labelled
        daily bar carries. Pass ``instants=True`` for intraday observations:
        with hours declared, the time of day is then checked against the
        session window too, so pre-market prints and bars past an early close
        are rejected rather than silently traded on.
        """
        series = timestamps if isinstance(timestamps, pl.Series) else pl.Series(list(timestamps))
        if series.is_empty():
            return
        if isinstance(series.dtype, pl.Datetime):
            moments = series
        else:
            moments = normalize_date_column(pl.DataFrame({"date": series}))["date"]
        dates = moments.cast(pl.Date)
        off_calendar = dates.filter(~dates.is_in(self.sessions.cast(pl.Date).implode())).unique().sort()
        if len(off_calendar):
            examples = [value.isoformat() for value in off_calendar.head(5).to_list()]
            raise ValueError(f"{label} contain dates outside calendar {self.name}: {examples}")
        if not instants:
            return
        self._require_hours()
        outside = [moment for moment in moments.to_list() if moment is not None and not self.is_open(moment)]
        if outside:
            examples = [moment.isoformat() for moment in sorted(outside)[:5]]
            raise ValueError(f"{label} fall outside session hours on calendar {self.name}: {examples}")


def infer_calendar_from_index(sessions, *, name: str = "inferred") -> TradingCalendar:
    """Build a calendar from a date-like sequence or column."""
    return TradingCalendar.from_sessions(sessions, name=name)


def infer_session_hours(
    timestamps,
    *,
    zone: str,
    name: str = "inferred",
    bar_length: timedelta = timedelta(0),
) -> TradingCalendar:
    """Derive sessions and their hours from what the timestamps actually span.

    This reports the data's window, which is not always the exchange's. Measured
    against two years of real minute bars: a 1:00pm NYSE close does **not** show
    up here, because post-market trading continues past it and looks identical
    in OHLCV -- SPY carried 4.9 million shares after the early close on
    2023-11-24. Early closes have to come from an exchange calendar; pass them
    to :meth:`TradingCalendar.with_hours` as overrides.

    What the outliers do reveal is missing data. On the same sample the sessions
    that differed from the modal window were 2023-03-10 (last bar 14:29) and the
    two days after it -- the vendor's gap during the SVB collapse, not a market
    event.

    The regular window is the most common (open, close) pair across sessions and
    every session that differs becomes an override.

    ``bar_length`` converts start-labelled bars to the instant the last bar
    completes; leave it at zero for timestamps that are already instants. A
    session observed only once cannot show a window and is rejected -- one bar
    says nothing about when the market closed.
    """
    series = timestamps if isinstance(timestamps, pl.Series) else pl.Series("timestamp", list(timestamps))
    if series.dtype == pl.String:
        series = series.str.to_datetime(strict=False)
    series = series.drop_nulls()
    if series.is_empty():
        raise ValueError("cannot infer session hours from no timestamps")
    if isinstance(series.dtype, pl.Datetime) and series.dtype.time_zone is not None:
        series = series.dt.convert_time_zone(zone).dt.replace_time_zone(None)
    frame = pl.DataFrame({"timestamp": series}).with_columns(pl.col("timestamp").cast(pl.Datetime("us")))
    frame = frame.with_columns(pl.col("timestamp").dt.truncate("1d").alias("session"))
    windows = frame.group_by("session").agg(
        pl.col("timestamp").min().alias("first"),
        pl.col("timestamp").max().alias("last"),
        pl.len().alias("observations"),
    ).sort("session")
    thin = windows.filter(pl.col("observations") < 2)["session"].to_list()
    if thin:
        examples = [value.date().isoformat() for value in thin[:5]]
        raise ValueError(f"sessions with a single observation cannot define hours: {examples}")
    pairs = [
        (row["session"], row["first"].time(), (row["last"] + bar_length).time())
        for row in windows.iter_rows(named=True)
    ]
    counts = Counter((opened, closed) for _, opened, closed in pairs)
    regular = max(counts, key=lambda key: (counts[key], key[1]))
    overrides = {
        session.date(): SessionWindow(opened, closed)
        for session, opened, closed in pairs
        if (opened, closed) != regular
    }
    calendar = TradingCalendar.from_sessions(windows["session"], name=name)
    return calendar.with_hours(regular[0], regular[1], zone=zone, overrides=overrides)


def _normalize_sessions(sessions) -> pl.Series:
    if isinstance(sessions, TradingCalendar):
        return sessions.sessions
    if isinstance(sessions, pl.Series):
        series = sessions
    elif isinstance(sessions, pl.DataFrame):
        series = sessions[sessions.columns[0]]
    else:
        series = pl.Series("session", list(sessions))
    if series.dtype == pl.String:
        series = series.str.to_datetime(strict=False)
    else:
        series = series.cast(pl.Datetime("us"), strict=False)
    # Normalize to midnight, drop duplicates, sort.
    series = series.dt.truncate("1d").unique().sort().rename("session")
    if series.is_empty():
        raise ValueError("sessions must not be empty")
    return series
