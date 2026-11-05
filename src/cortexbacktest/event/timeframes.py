"""Timeframe resampling and replay helpers."""

from __future__ import annotations

from copy import replace
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Iterable, Iterator, Literal

from ..contracts import Bar

if TYPE_CHECKING:
    from ..data.calendar import TradingCalendar


def resample_bars(
    bars: Iterable[Bar], timeframe: str, *,
    timestamp_label: Literal["end", "start"] = "end",
    source_timeframe: str | None = None,
    calendar: "TradingCalendar | None" = None,
) -> list[Bar]:
    """Aggregate start-labelled observations; publish at bucket completion.

    ``end`` is the safe default for event feeds. ``start`` is a chart/export
    label only and must not be used as the strategy availability timestamp.
    If source bars cover intervals, declare source_timeframe to reject bars
    straddling a target boundary. Missing source observations are not invented.

    Pass a ``calendar`` carrying session hours to stamp the last bucket of a
    session at the close instead of at a wall-clock instant past it: the 15:45
    ET half-hour bucket completes at 16:00, not 16:15, and on an early close at
    13:00, not at 16:00.
    """
    if timestamp_label not in {"start", "end"}:
        raise ValueError("timestamp_label must be 'start' or 'end'")
    buckets: dict[tuple[str, datetime], Bar] = {}
    for bar in sorted(bars, key=lambda item: (_instant(item.timestamp), item.symbol)):
        start = bucket_timestamp(bar.timestamp, timeframe)
        end = bucket_end(bar.timestamp, timeframe, calendar=calendar)
        if source_timeframe and _instant(_source_available_at(bar.timestamp, source_timeframe)) > _instant(end):
            raise ValueError("source bar straddles target timeframe boundary")
        key = (bar.symbol.upper(), _instant(start))
        current = buckets.get(key)
        buckets[key] = _merge_bar(current, replace(bar, symbol=bar.symbol.upper(), timestamp=start))
    output = [replace(bar, timestamp=bucket_end(bar.timestamp, timeframe, calendar=calendar))
              if timestamp_label == "end" else bar for bar in buckets.values()]
    return sorted(output, key=lambda bar: (_instant(bar.timestamp), bar.symbol))


def replay_bars(bars: Iterable[Bar], timeframe: str, *, source_timeframe: str | None = None) -> Iterator[Bar]:
    """Yield incremental higher-timeframe bars after each source bar.

    This mirrors Backtrader-style replay semantics at a lightweight data layer:
    every incoming observation updates the bucket and emits a partial aggregate
    stamped at its availability time, never backdated to the bucket start.
    Without source_timeframe, input timestamps must already be availability
    times (e.g. ticks). For start-labelled interval bars, supply source_timeframe.
    Partial bars are cumulative: do not use their volume as independent fills.
    """
    buckets: dict[tuple[str, datetime], Bar] = {}
    for bar in sorted(bars, key=lambda item: (_instant(item.timestamp), item.symbol)):
        start = bucket_timestamp(bar.timestamp, timeframe)
        key = (bar.symbol.upper(), _instant(start))
        current = buckets.get(key)
        updated = _merge_bar(current, replace(bar, symbol=bar.symbol.upper(), timestamp=start))
        buckets[key] = updated
        available_at = _source_available_at(bar.timestamp, source_timeframe) if source_timeframe else bar.timestamp
        if _instant(available_at) > _instant(bucket_end(bar.timestamp, timeframe)):
            raise ValueError("source bar straddles target timeframe boundary")
        yield replace(updated, timestamp=available_at)


def _instant(timestamp: datetime) -> datetime:
    return timestamp.astimezone(UTC) if timestamp.tzinfo else timestamp


def _source_available_at(timestamp: datetime, timeframe: str) -> datetime:
    if _instant(timestamp) != _instant(bucket_timestamp(timestamp, timeframe)):
        raise ValueError("source interval timestamps must align with source_timeframe buckets")
    return bucket_end(timestamp, timeframe)


def bucket_end(timestamp: datetime, timeframe: str, *, calendar: "TradingCalendar | None" = None) -> datetime:
    """Exclusive end of a wall-clock bucket; intraday durations use elapsed time.

    With a session-aware ``calendar``, a bucket that would end after the close
    ends at the close instead. Without one the bucket is pure wall-clock
    arithmetic capped at midnight, which is all a calendar-free caller can
    honestly say.
    """
    start = bucket_timestamp(timestamp, timeframe)
    value = timeframe.strip().lower()
    if value in {"d", "1d", "day", "daily"}:
        return start + timedelta(days=1)
    if value in {"w", "1w", "week", "weekly"}:
        return start + timedelta(days=7)
    minutes = int(value[:-3]) if value.endswith("min") else int(value[:-1]) * (60 if value.endswith("h") else 1)
    end = _instant(start) + timedelta(minutes=minutes)
    midnight = (start + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0, fold=0)
    end = min(end, _instant(midnight))
    end = end.astimezone(timestamp.tzinfo) if timestamp.tzinfo else end
    if calendar is not None and calendar.has_hours:
        end = calendar.clamp_to_close(timestamp, end)
    return end


def bucket_timestamp(timestamp: datetime, timeframe: str) -> datetime:
    value = timeframe.strip().lower()
    if value in {"d", "1d", "day", "daily"}:
        return timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
    if value in {"w", "1w", "week", "weekly"}:
        start = timestamp - timedelta(days=timestamp.weekday())
        return start.replace(hour=0, minute=0, second=0, microsecond=0)
    if value.endswith(("min", "m")):
        minutes = int(value[:-3] if value.endswith("min") else value[:-1])
        return _minute_bucket(timestamp, minutes)
    if value.endswith("h"):
        hours = int(value[:-1])
        if hours < 1 or hours > 24:
            raise ValueError("hour timeframe must be in [1, 24]")
        hour = (timestamp.hour // hours) * hours
        return timestamp.replace(hour=hour, minute=0, second=0, microsecond=0)
    raise ValueError(f"unsupported timeframe: {timeframe}")


def _minute_bucket(timestamp: datetime, minutes: int) -> datetime:
    if minutes < 1 or minutes > 24 * 60:
        raise ValueError("minute timeframe must be in [1, 1440]")
    total = timestamp.hour * 60 + timestamp.minute
    bucket = (total // minutes) * minutes
    return timestamp.replace(hour=bucket // 60, minute=bucket % 60, second=0, microsecond=0)


def _merge_bar(current: Bar | None, bar: Bar) -> Bar:
    if current is None:
        return bar
    volume = None
    if current.volume is not None or bar.volume is not None:
        volume = (current.volume or 0.0) + (bar.volume or 0.0)
    return Bar(
        symbol=current.symbol,
        timestamp=current.timestamp,
        open=current.open,
        high=max(current.high, bar.high),
        low=min(current.low, bar.low),
        close=bar.close,
        volume=volume,
        adj_close=bar.adj_close,
        vwap=_merged_vwap(current, bar),
        open_interest=bar.open_interest,
        halted=current.halted and bar.halted,
        limit_up=bar.limit_up,
        limit_down=bar.limit_down,
    )


def _merged_vwap(current: Bar, bar: Bar) -> float | None:
    if current.vwap is None or bar.vwap is None or current.volume is None or bar.volume is None:
        return None
    total_volume = current.volume + bar.volume
    if total_volume <= 0:
        return None
    return (current.vwap * current.volume + bar.vwap * bar.volume) / total_volume
