"""Columnar market-data feed helpers for event backtests."""

from __future__ import annotations

from copy import replace
from dataclasses import dataclass
from datetime import datetime
from itertools import chain, groupby, repeat
import math
from operator import attrgetter
from typing import Iterable, Iterator, NamedTuple, Sequence

import numpy as np
import polars as pl

from ..contracts import Bar
from ..data.backend import to_backend
from .manifest import BarDataFingerprint, fingerprint_bars

# C-level key functions: the lambda equivalents build a frame (and, for the
# sort key, a tuple) per bar, which is measurable at millions of bars.
_BAR_SORT_KEY = attrgetter("timestamp", "symbol")
_BAR_TIMESTAMP = attrgetter("timestamp")


class MarketSlice(NamedTuple):
    """Bars for one timestamp.

    A NamedTuple rather than a frozen dataclass: a single-symbol feed builds
    one of these per bar, and frozen-dataclass construction goes through
    object.__setattr__ twice per instance — several times slower at millions
    of bars for the same immutable (timestamp, bars) shape.
    """

    timestamp: datetime
    bars: tuple[Bar, ...]


class ColumnarBarFeed:
    """Pre-grouped event feed that preserves the public `Bar` contract.

    The feed stores sorted market slices once, so repeated runs and execution
    sweeps do not re-sort and re-group a raw list of bars inside the event loop.

    Bar objects are materialized once and reused on every iteration pass by
    default (`cache=True` on `from_polars`) — the right trade for an execution
    sweep or a parameter search, which iterate the same feed hundreds of times.
    `cache=False` instead keeps the ingested data as plain arrays (numpy, one
    per field — no per-bar Python object, no per-field boxed float) and builds
    a `MarketSlice`'s `Bar` objects lazily, one slice at a time, during
    iteration, without retaining them — the right trade for a single very
    large pass (a 10M-bar run once) where standing memory matters more than a
    second iteration being cheap, because a second iteration re-pays the
    materialization cost `cache=True` would have avoided.
    """

    def __init__(self, slices: Sequence[MarketSlice], *, columns: "_Columns | None" = None) -> None:
        self._columns = columns
        if columns is not None:
            self._slices = ()
            self._single_bars = None
        elif slices and all(len(market_slice.bars) == 1 for market_slice in slices):
            self._slices = ()
            self._single_bars = tuple(market_slice.bars[0] for market_slice in slices)
        else:
            self._slices = tuple(slices)
            self._single_bars = None
        self._data_fingerprint: BarDataFingerprint | None = None

    @classmethod
    def _from_single_bars(cls, bars: Iterable[Bar]) -> "ColumnarBarFeed":
        feed = cls(())
        feed._single_bars = tuple(bars)
        return feed

    @classmethod
    def _from_columns(cls, columns: "_Columns") -> "ColumnarBarFeed":
        return cls((), columns=columns)

    @classmethod
    def from_bars(cls, bars: Iterable[Bar], *, assume_sorted: bool = False) -> "ColumnarBarFeed":
        bars = list(bars)
        if bars:
            # A naive stamp is an unverified clock label, an aware one an instant;
            # one feed must not interleave them. Checked before sorting so the
            # rule does not depend on `assume_sorted` or on sort() happening to
            # compare a mixed pair.
            naive = bars[0].timestamp.tzinfo is None
            for bar in bars:
                if (bar.timestamp.tzinfo is None) is not naive:
                    raise ValueError(
                        "bar feed mixes naive and timezone-aware timestamps; "
                        f"{bar.symbol} at {bar.timestamp!r} differs from the first bar"
                    )
        if not assume_sorted:
            bars.sort(key=_BAR_SORT_KEY)
        for index, bar in enumerate(bars):
            _validate_bar(bar)
            # Broker order matching and strategy slices use upper-case symbols;
            # normalize here so lower-case bar symbols cannot silently never fill.
            # isupper() is false for uncased symbols ("600519"), so those fall
            # through to the allocation to be checked properly.
            symbol = bar.symbol
            if not symbol.isupper():
                upper_symbol = symbol.upper()
                if upper_symbol != symbol:
                    bars[index] = replace(bar, symbol=upper_symbol)
        if all(
            bars[index - 1].timestamp != bars[index].timestamp
            for index in range(1, len(bars))
        ):
            return cls._from_single_bars(bars)
        slices = [
            MarketSlice(timestamp=timestamp, bars=tuple(group))
            for timestamp, group in groupby(bars, key=_BAR_TIMESTAMP)
        ]
        return cls(slices)

    @classmethod
    def from_polars(
        cls,
        frame,
        *,
        date_col: str = "date",
        symbol_col: str = "symbol",
        open_col: str = "open",
        high_col: str = "high",
        low_col: str = "low",
        close_col: str = "close",
        volume_col: str = "volume",
        adj_close_col: str = "adj_close",
        vwap_col: str = "vwap",
        cache: bool = True,
    ) -> "ColumnarBarFeed":
        data = to_backend(frame, backend="polars")
        required = {date_col, symbol_col, open_col, high_col, low_col, close_col}
        missing = required.difference(data.columns)
        if missing:
            raise ValueError(f"Missing required bar columns: {sorted(missing)}")
        optional = [
            pl.col(name) if name in data.columns else pl.lit(None).alias(name)
            for name in (volume_col, adj_close_col, vwap_col, "open_interest", "halted", "limit_up", "limit_down")
        ]
        # Select a fixed positional layout before crossing into Python. Named
        # rows allocate one dict per bar and the old path then retained a second
        # full `bars` list before grouping it into slices.
        sorted_data = (
            data.select(
                date_col,
                # Normalize in Polars, once per column, rather than in Python
                # once per row. The row loop below then only builds Bars: no
                # str().upper(), no float(), no per-row None test on a value the
                # cast has already settled.
                pl.col(symbol_col).cast(pl.Utf8).str.to_uppercase(),
                pl.col(open_col).cast(pl.Float64),
                pl.col(high_col).cast(pl.Float64),
                pl.col(low_col).cast(pl.Float64),
                pl.col(close_col).cast(pl.Float64),
                *optional,
            )
            .sort([date_col, symbol_col])
            .rename(
                {
                    symbol_col: "symbol",
                    open_col: "open",
                    high_col: "high",
                    low_col: "low",
                    close_col: "close",
                    volume_col: "volume",
                    adj_close_col: "adj_close",
                    vwap_col: "vwap",
                },
                strict=False,
            )
            .with_columns(
                pl.col("volume").cast(pl.Float64),
                pl.col("adj_close").cast(pl.Float64),
                pl.col("vwap").cast(pl.Float64),
                pl.col("open_interest").cast(pl.Float64),
                pl.col("halted").cast(pl.Boolean).fill_null(False),
                pl.col("limit_up").cast(pl.Float64),
                pl.col("limit_down").cast(pl.Float64),
            )
        )
        if not cache:
            return cls._from_columns(_columns_from_sorted_frame(sorted_data, date_col=date_col))
        # Unique timestamps need only the Bar tuple used by the engine's fast
        # path. Avoid allocating one temporary list, tuple and MarketSlice per
        # row only for __init__ to discard all of them again.
        # One vectorized pass over the columns instead of a _validate_bar call
        # per constructed Bar. Same acceptance decision -- it is the same check
        # the uncached path already ran -- reached without touching the objects.
        _validate_sorted_frame(sorted_data, date_col=date_col)
        single = sorted_data[date_col].n_unique() == sorted_data.height
        # Whether timestamps need converting is a property of the column, not of
        # each row; deciding it once removes a hasattr per bar.
        rows = sorted_data.iter_rows(named=False)
        first = next(rows, None)
        if first is None:
            # An empty frame has no repeated timestamps, so it is the unique
            # case, and callers rely on single_bars being an empty tuple there.
            return cls._from_single_bars([])
        if hasattr(first[0], "to_pydatetime"):
            rows = (( row[0].to_pydatetime(), *row[1:]) for row in rows)
            first = (first[0].to_pydatetime(), *first[1:])

        make = Bar
        if single:
            return cls._from_single_bars(make(row[1], row[0], *row[2:]) for row in chain((first,), rows))

        slices: list[MarketSlice] = []
        slice_bars: list[Bar] = [make(first[1], first[0], *first[2:])]
        current_timestamp = first[0]
        for row in rows:
            timestamp = row[0]
            if timestamp != current_timestamp:
                slices.append(MarketSlice(current_timestamp, tuple(slice_bars)))
                slice_bars = []
                current_timestamp = timestamp
            slice_bars.append(make(row[1], timestamp, *row[2:]))
        slices.append(MarketSlice(current_timestamp, tuple(slice_bars)))
        return cls(slices)

    @property
    def single_bars(self) -> tuple[Bar, ...] | None:
        """Immutable single-bar slices, or None when grouped iteration is required."""
        return self._single_bars

    def __iter__(self) -> Iterator[MarketSlice]:
        if self._columns is not None:
            return self._columns.iter_slices()
        if self._single_bars is not None:
            return map(_single_bar_slice, self._single_bars)
        return iter(self._slices)

    def __len__(self) -> int:
        if self._columns is not None:
            return len(self._columns.slice_starts)
        if self._single_bars is not None:
            return len(self._single_bars)
        return len(self._slices)

    def data_fingerprint(self) -> BarDataFingerprint:
        fingerprint = self._data_fingerprint
        if fingerprint is None:
            fingerprint = fingerprint_bars(bar for group in self for bar in group.bars)
            self._data_fingerprint = fingerprint
        return fingerprint

    @property
    def bars(self) -> list[Bar]:
        """Every bar, materialized. For a cache=False feed this pays the same
        per-bar construction cost from_polars(cache=True) would have paid
        eagerly — deferred here to the point something actually asks for
        Bar objects, per this feed's whole reason to exist."""
        if self._columns is not None:
            return [bar for market_slice in self._columns.iter_slices() for bar in market_slice.bars]
        if self._single_bars is not None:
            return list(self._single_bars)
        return [bar for market_slice in self._slices for bar in market_slice.bars]


@dataclass(slots=True)
class _Columns:
    """Row-aligned numpy arrays backing a cache=False ColumnarBarFeed.

    One array per Bar field, plus slice_starts (the row index each unique
    timestamp group begins at) and slice_timestamps (one entry per group,
    since every bar in a group shares its timestamp — storing it once per
    group rather than once per row). Nullable fields (volume, adj_close,
    vwap, open_interest, limit_up, limit_down) are float64 with NaN standing
    in for None; Bar's contract keeps those as `float | None`, so bar_at
    converts NaN back to None on the way out.
    """

    length: int
    slice_starts: np.ndarray
    slice_timestamps: list[datetime]
    symbol: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    adj_close: np.ndarray
    vwap: np.ndarray
    open_interest: np.ndarray
    halted: np.ndarray
    limit_up: np.ndarray
    limit_down: np.ndarray

    def bar_at(self, index: int, timestamp: datetime) -> Bar:
        return Bar(
            symbol=self.symbol[index],
            timestamp=timestamp,
            open=float(self.open[index]),
            high=float(self.high[index]),
            low=float(self.low[index]),
            close=float(self.close[index]),
            volume=_nan_to_none(self.volume[index]),
            adj_close=_nan_to_none(self.adj_close[index]),
            vwap=_nan_to_none(self.vwap[index]),
            open_interest=_nan_to_none(self.open_interest[index]),
            halted=bool(self.halted[index]),
            limit_up=_nan_to_none(self.limit_up[index]),
            limit_down=_nan_to_none(self.limit_down[index]),
        )

    def iter_slices(self) -> Iterator[MarketSlice]:
        starts = self.slice_starts
        group_count = len(starts)
        for group_index in range(group_count):
            start = int(starts[group_index])
            end = int(starts[group_index + 1]) if group_index + 1 < group_count else self.length
            timestamp = self.slice_timestamps[group_index]
            if end - start < 32:
                yield MarketSlice(timestamp, tuple(self.bar_at(row, timestamp) for row in range(start, end)))
                continue
            # Convert contiguous arrays in C once per market slice instead of
            # indexing and boxing 13 NumPy scalars for every Bar. Temporaries
            # live for one slice only, preserving the uncached memory bound.
            rows = zip(
                self.symbol[start:end].tolist(), repeat(timestamp),
                self.open[start:end].tolist(), self.high[start:end].tolist(),
                self.low[start:end].tolist(), self.close[start:end].tolist(),
                _nullable_slice(self.volume, start, end),
                _nullable_slice(self.adj_close, start, end),
                _nullable_slice(self.vwap, start, end),
                _nullable_slice(self.open_interest, start, end),
                self.halted[start:end].tolist(),
                _nullable_slice(self.limit_up, start, end),
                _nullable_slice(self.limit_down, start, end),
            )
            yield MarketSlice(timestamp, tuple(map(Bar._make, rows)))


def _nullable_slice(values: np.ndarray, start: int, end: int) -> Iterable[float | None]:
    chunk = values[start:end]
    nulls = np.isnan(chunk)
    if nulls.all():
        return repeat(None, end - start)
    boxed = chunk.tolist()
    if nulls.any():
        for index in np.flatnonzero(nulls):
            boxed[index] = None
    return boxed


def _nan_to_none(value: float) -> float | None:
    return None if math.isnan(value) else float(value)


def _columns_from_sorted_frame(frame: pl.DataFrame, *, date_col: str) -> _Columns:
    """Build a _Columns from a frame already sorted by (date_col, "symbol")
    and renamed to the fixed field names from_polars uses internally.

    Validation is shared with the cached path through _validate_sorted_frame,
    so the two cache modes cannot drift on which observations they accept.
    """
    length = frame.height
    open_, high, low, close = _validate_sorted_frame(frame, date_col=date_col)

    timestamp_values = frame[date_col].to_numpy()
    starts = np.r_[0, np.flatnonzero(timestamp_values[1:] != timestamp_values[:-1]) + 1] if length else np.array([], dtype=np.int64)
    slice_timestamps = [_as_datetime(value) for value in frame[date_col].gather(starts).to_list()]
    return _Columns(
        length=length,
        slice_starts=np.asarray(starts, dtype=np.int64),
        slice_timestamps=slice_timestamps,
        symbol=frame["symbol"].to_numpy(),
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=frame["volume"].to_numpy(),
        adj_close=frame["adj_close"].to_numpy(),
        vwap=frame["vwap"].to_numpy(),
        open_interest=frame["open_interest"].to_numpy(),
        halted=frame["halted"].to_numpy(),
        limit_up=frame["limit_up"].to_numpy(),
        limit_down=frame["limit_down"].to_numpy(),
    )


def _as_datetime(value) -> datetime:
    return value.to_pydatetime() if hasattr(value, "to_pydatetime") else value


def _validate_sorted_frame(frame: pl.DataFrame, *, date_col: str) -> tuple[np.ndarray, ...]:
    """Check every bar in one vectorized pass; return the OHLC arrays.

    Both cache modes need the same answer, and the column-at-a-time check is
    far cheaper than one `_validate_bar` call per constructed Bar. Only a
    failure pays to materialize the single offending Bar, so the error message
    stays exactly the one `_validate_bar` produces.
    """
    open_ = frame["open"].to_numpy()
    high = frame["high"].to_numpy()
    low = frame["low"].to_numpy()
    close = frame["close"].to_numpy()
    bad = _first_invalid_row(
        open_, high, low, close, frame["volume"].to_numpy(), frame["vwap"].to_numpy(),
        volume_null=frame["volume"].is_null().to_numpy(),
        vwap_null=frame["vwap"].is_null().to_numpy(),
    )
    if bad is not None:
        _validate_bar(
            Bar(
                symbol=str(frame["symbol"][bad]),
                timestamp=_as_datetime(frame[date_col][bad]),
                open=float(open_[bad]),
                high=float(high[bad]),
                low=float(low[bad]),
                close=float(close[bad]),
                volume=_optional_float(frame["volume"][bad]),
                vwap=_optional_float(frame["vwap"][bad]),
            )
        )
        raise AssertionError("vectorized bar validation disagreed with _validate_bar")  # pragma: no cover
    return open_, high, low, close


def _first_invalid_row(
    open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray, vwap: np.ndarray,
    *, volume_null: np.ndarray, vwap_null: np.ndarray,
) -> int | None:
    lower_body = np.minimum(open_, close)
    upper_body = np.maximum(open_, close)
    finite_positive = (
        np.isfinite(open_) & (open_ > 0)
        & np.isfinite(high) & (high > 0)
        & np.isfinite(low) & (low > 0)
        & np.isfinite(close) & (close > 0)
    )
    range_ok = (low <= lower_body) & (high >= upper_body)
    # NumPy represents both Polars null and a real NaN as NaN. Only null is
    # missing data; preserve the distinction so cache mode cannot change which
    # observations are accepted by the execution engine.
    volume_ok = volume_null | (np.isfinite(volume) & (volume >= 0))
    vwap_ok = vwap_null | (np.isfinite(vwap) & (low <= vwap) & (vwap <= high))
    valid = finite_positive & range_ok & volume_ok & vwap_ok
    bad_rows = np.flatnonzero(~valid)
    return int(bad_rows[0]) if bad_rows.size else None


def market_slices_from_bars(bars: Iterable[Bar] | ColumnarBarFeed, *, assume_sorted: bool = False) -> ColumnarBarFeed:
    if isinstance(bars, ColumnarBarFeed):
        return bars
    return ColumnarBarFeed.from_bars(bars, assume_sorted=assume_sorted)


def _optional_float(value) -> float | None:
    if value is None:
        return None
    return float(value)


def _single_bar_slice(bar: Bar) -> MarketSlice:
    return MarketSlice(bar.timestamp, (bar,))


def _validate_bar(bar: Bar) -> None:
    """Reject impossible OHLCV observations before they reach execution logic.

    The happy path allocates nothing: this runs once per bar, and the dict +
    list comprehension it used to build cost ~5% of a large ingest. Names are
    only assembled once a bar has already failed.
    """
    open_ = bar.open
    high = bar.high
    low = bar.low
    close = bar.close
    isfinite = math.isfinite
    if not (isfinite(open_) and open_ > 0 and isfinite(high) and high > 0
            and isfinite(low) and low > 0 and isfinite(close) and close > 0):
        prices = {"open": open_, "high": high, "low": low, "close": close}
        invalid = [name for name, value in prices.items() if not isfinite(value) or value <= 0]
        raise ValueError(f"bar {bar.symbol} has non-finite or non-positive prices: {invalid}")
    if low > (open_ if open_ < close else close) or high < (open_ if open_ > close else close):
        raise ValueError(f"bar {bar.symbol} has OHLC values outside its high/low range")
    volume = bar.volume
    if volume is not None and (not isfinite(volume) or volume < 0):
        raise ValueError(f"bar {bar.symbol} has invalid volume")
    vwap = bar.vwap
    if vwap is not None and (not isfinite(vwap) or not low <= vwap <= high):
        raise ValueError(f"bar {bar.symbol} has VWAP outside its high/low range")
