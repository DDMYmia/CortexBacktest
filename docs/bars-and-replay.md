# Bars, sessions and large replays

`Bar.timestamp` is a `datetime`, and the caller declares the resolution. A daily
date can be a session label. Intraday observation and availability times must
follow the data contract: the timestamp type alone does not prove there is no
look-ahead.

## Resolution checks

`validate_bars(frame, *, calendar=None, resolution=...)`:

| resolution | Checks |
| --- | --- |
| `"any"` (default) | Nulls, duplicate keys, OHLC consistency, non-negative finite volume; dates, when a calendar is given |
| `"session"` | Also rejects more than one observation per symbol per session |
| `"intraday"` | When the calendar has session hours, also checks that observations fall inside the session (the close itself excluded) |

`load_backtest_dataset` defaults to `resolution="session"`. Intraday research
must pass `resolution="intraday"` and a calendar with session hours.
`bars_to_events` defaults to `"any"`; pass a calendar and `"intraday"` for the
strict session check.

```python
calendar = TradingCalendar.from_exchange("XNYS", "2024-01-01", "2024-12-31")
dataset = load_backtest_dataset(
    bars=minute_bars, factors=factors, calendar=calendar, resolution="intraday",
)
```

With several rows on one local date, `returns_matrix` sets the first row of each
session to null, `overnight_returns_matrix` keeps only the first row's overnight
change, and `forward_returns_matrix` sets horizons that cross a session to null.
Called directly, these functions group by local date and do not check session
hours; `tests/test_session_resolution.py` covers the full path and half days.

## Session hours

A date-only calendar knows which dates are sessions. Queries at a point in time
need `hours`, and `hours` needs `zone`. A naive datetime is read as wall-clock
time in the calendar's zone; an aware datetime is converted to that zone first.

```python
from datetime import date, datetime, time
from cortexbacktest.data import SessionWindow, TradingCalendar

calendar = TradingCalendar.business_days("2024-11-25", "2024-11-29").with_hours(
    time(9, 30), time(16, 0), zone="America/New_York",
    overrides={date(2024, 11, 29): SessionWindow(time(9, 30), time(13, 0))},
)
assert calendar.session_close("2024-11-29").hour == 13
assert not calendar.is_open(datetime(2024, 11, 29, 13))
```

`business_days` only excludes weekends, and the example declares its hours by
hand, so it is not an exchange holiday calendar. `SessionWindow` requires
open < close and does not support sessions that cross midnight.

| Source | Evidence and limits |
| --- | --- |
| `TradingCalendar.from_exchange` | Regular hours and exceptions, early closes included, from the exchange calendar. Needs the optional `calendar` extra (`pandas_market_calendars`). |
| `infer_session_hours(timestamps, zone=..., bar_length=...)` | Infers the window from the first and last observations; `bar_length` is the length of the last bar, labelled by its start. It cannot tell extended hours, missing data and real exchange hours apart. |

Both return a regular window plus date exceptions, from different evidence. For
the exchange's close, use an exchange calendar or explicit hours with a source;
an observed window does not establish exchange policy.

## Buckets and availability

`bucket_end` clips an intraday observation inside the session to the close when
a `calendar` is given; outside the session it does not guess a close.
`resample_bars` recomputes the label from the bucket start, so a bucket that
starts before the open can lose the early-close clip: an 8h bucket of a 09:30
observation is still labelled 16:00 on a 13:00 early-close day. This edge is not
fixed, so an aggregated label is not guaranteed to fall inside the session.
Without a calendar, intraday buckets follow the wall clock, advance by their real
length and stop at midnight; daily and weekly buckets have their own rules.

```python
from cortexbacktest.event.timeframes import bucket_end

assert bucket_end(datetime(2024, 11, 28, 15), "3h").hour == 18
assert bucket_end(datetime(2024, 11, 28, 15), "3h", calendar=calendar).hour == 16
assert bucket_end(datetime(2024, 11, 29, 12), "2h", calendar=calendar).hour == 13
```

Aggregated labels, source-bar availability times and option publication times
each keep their own contract; see [execution](execution.md#time-semantics).
Calendar checks do not replace them.

## Large minute replays

`ParquetBarFeed` reads time-partitioned Parquet in batches and hands the event
engine one complete market time slice at a time. Unlike
`from_polars(cache=False)`, it does not collect, sort and hold the whole input in
memory. It is the data path for decade-long, thousand-symbol backtests; it is not
a capacity certificate for a real strategy at that scale.


```python
from cortexbacktest.engine import BacktestEngine
from cortexbacktest.event.parquet_feed import ParquetBarFeed

feed = ParquetBarFeed(
    ["bars/2020-01.parquet", "bars/2020-02.parquet"],
    batch_rows=65_536,
    date_col="date",
)
result = BacktestEngine(
    snapshot_frequency="session",
    result_retention="summary",
    build_manifest=True,
).run(feed, strategy)
```

- Paths must be in increasing time order, and timestamps inside each file must
  not decrease. Use one normalized time zone and precision. Columns match the
  `ColumnarBarFeed.from_polars` defaults; only the time column name can change.
- One minute can span several batches, row groups or adjacent files. The
  strategy sees it only once it is complete, with symbols upper-cased and sorted;
  a duplicate symbol in one minute is rejected.
- Files are concatenated in time order by default. Symbol partitions that are
  each sorted can be merged with `merge_sorted=True`, bounded by
  `max_open_files`. This does not sort unordered files, and memory then includes
  the pending slice of each input.
- Each iteration reopens the files, so the input must not change during the
  backtest and the fingerprint pass; the fingerprint is not cached across runs.
  The default `data_validation="preflight"` adds a scan before the run;
  `"stream"` folds validation and the manifest digest into the replay, so an error
  can surface after earlier events have run.
- The working set grows with the batch size and the largest time slice. Parquet
  metadata, decoder buffers, the strategy, positions, open orders and retained
  results are extra. Very large row groups and very many small files need their
  own measurement.
- `len(feed)` and `feed.bars` are not supported, because they would materialize
  the input; count by iterating. `PartitionedBarFeed` collects whole partitions,
  supports `len` / `bars` and caches its fingerprint, so the batch-size and
  fingerprint guarantees here do not apply to it.
- `summary` retention does not keep the full fill history. Some analyzers make
  the engine retain records again; measure memory separately for full analyzers,
  per-minute snapshots or strategies that accumulate their own history.
  `snapshot_frequency="none"` suits capacity tests and jobs that need no equity
  curve, not full performance analysis.

Tests cover equal net value and fills across batches and files, fingerprint
equality, repeated iteration, and rejection of unordered, empty, invalid and
duplicate-symbol input. `benchmarks/parquet_streaming.py` is a reproducible
empty-strategy capacity run; it is not a capacity certificate for a real
strategy. For planning, 10 years × 1,000 symbols of 390 one-minute bars a day is
about 982.8M rows.
