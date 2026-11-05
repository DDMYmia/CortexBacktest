"""Replay time-partitioned Parquet without collecting the complete dataset."""

from __future__ import annotations

import heapq
from datetime import datetime
from pathlib import Path
from typing import Generator, Sequence, cast

import polars as pl
import pyarrow.parquet as pq

from ..contracts import Bar
from .feed import ColumnarBarFeed, MarketSlice
from .manifest import BarDataFingerprint, fingerprint_bars


class ParquetBarFeed(ColumnarBarFeed):
    """Repeatable bounded-batch feed for chronologically ordered Parquet files.

    Files must be supplied in time order; timestamps within each file must be
    nondecreasing. A timestamp may cross batches/files, and is emitted exactly
    once with symbols sorted. Duplicate normalized symbols are rejected.
    Memory is proportional to batch_rows plus the largest timestamp group,
    excluding the Parquet reader's metadata/decoder buffers and engine state.
    Input files must remain immutable throughout all passes. Validation is lazy:
    a bad later batch can raise after earlier simulation events have run.
    """

    def __init__(
        self,
        paths: str | Path | Sequence[str | Path],
        *,
        batch_rows: int = 65_536,
        date_col: str = "date",
        symbols: Sequence[str] | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        merge_sorted: bool = False,
        max_open_files: int = 1024,
    ) -> None:
        super().__init__(())
        if isinstance(batch_rows, bool) or not isinstance(batch_rows, int) or batch_rows < 1:
            raise ValueError("batch_rows must be a positive integer")
        self.paths = (Path(paths),) if isinstance(paths, (str, Path)) else tuple(map(Path, paths))
        if not self.paths:
            raise ValueError("at least one Parquet path is required")
        if start is not None and end is not None and start >= end:
            raise ValueError("start must precede end")
        if merge_sorted and len(self.paths) > max_open_files:
            raise ValueError("Too many merge inputs; compact files or raise max_open_files")
        self.symbols = None if symbols is None else sorted({s.upper() for s in symbols})
        self.start, self.end, self.merge_sorted = start, end, merge_sorted
        self.scan_stats = {}
        self.batch_rows = batch_rows
        self.date_col = date_col
        for path in self.paths:
            if not path.is_file():
                raise FileNotFoundError(path)

    def __len__(self) -> int:
        # Prevent Python list() from silently scanning the entire dataset just
        # for its length hint. Iteration and the event loop need no row count.
        raise TypeError("ParquetBarFeed has no cheap length; iterate to count slices")

    def __bool__(self) -> bool:
        return True

    @property
    def bars(self) -> list[Bar]:
        raise TypeError("ParquetBarFeed does not materialize all bars; iterate its slices")

    def data_fingerprint(self) -> BarDataFingerprint:
        # Do not reuse an in-memory fingerprint if the files changed between
        # runs. Like simulation, hashing is bounded-memory but reads every row.
        return fingerprint_bars(bar for group in self for bar in group.bars)

    def __iter__(self) -> Generator[MarketSlice, None, None]:
        self.scan_stats = {"row_groups_skipped": 0, "row_groups_read": 0, "rows_decoded": 0}
        if self.merge_sorted:
            yield from self._merged()
            return
        pending: list[Bar] = []
        timestamp: datetime | None = None
        last_input: datetime | None = None
        for path in self.paths:
            with pq.ParquetFile(path) as source:
                names = source.schema_arrow.names
                fields = [self.date_col, "symbol", "open", "high", "low", "close"]
                missing = set(fields).difference(names)
                if missing:
                    raise ValueError(f"Missing required bar columns in {path}: {sorted(missing)}")
                fields += [
                    name
                    for name in (
                        "volume",
                        "adj_close",
                        "vwap",
                        "open_interest",
                        "halted",
                        "limit_up",
                        "limit_down",
                    )
                    if name in names
                ]
                date_index = names.index(self.date_col)
                groups = []
                for index in range(source.num_row_groups):
                    stats = (
                        source.metadata.row_group(index)
                        .column(date_index)
                        .statistics
                    )
                    if (
                        stats is not None
                        and stats.has_min_max
                        and (
                            (self.start is not None and stats.max < self.start)
                            or (self.end is not None and stats.min >= self.end)
                        )
                    ):
                        self.scan_stats["row_groups_skipped"] += 1
                        continue
                    groups.append(index)
                self.scan_stats["row_groups_read"] += len(groups)
                for batch in source.iter_batches(
                    batch_size=self.batch_rows, columns=fields, row_groups=groups, use_threads=False
                ):
                    self.scan_stats["rows_decoded"] += batch.num_rows
                    frame = cast(pl.DataFrame, pl.from_arrow(batch))
                    if self.start is not None:
                        frame = frame.filter(pl.col(self.date_col) >= self.start)
                    if self.end is not None:
                        frame = frame.filter(pl.col(self.date_col) < self.end)
                    if self.symbols is not None:
                        frame = frame.filter(
                            pl.col("symbol").str.to_uppercase().is_in(self.symbols)
                        )
                    dates = frame[self.date_col]
                    if dates.null_count() or not dates.is_sorted():
                        raise ValueError(f"Parquet timestamps must be non-null and sorted: {path}")
                    if frame.is_empty():
                        continue
                    if last_input is not None and dates[0] < last_input:
                        raise ValueError(
                            f"Parquet timestamps decrease across batches/files: {path}"
                        )
                    last_input = dates[-1]
                    chunk = ColumnarBarFeed.from_polars(frame, date_col=self.date_col, cache=False)
                    for group in chunk:
                        if timestamp is not None and group.timestamp != timestamp:
                            yield self._finish(timestamp, pending)
                            pending = []
                        timestamp = group.timestamp
                        pending.extend(group.bars)
        if timestamp is not None:
            yield self._finish(timestamp, pending)

    def _merged(self) -> Generator[MarketSlice, None, None]:
        # Fixed decoder row budget shared across files; additional memory is
        # one pending timestamp group per input, decoder metadata and handles.
        feeds = [
            ParquetBarFeed(
                path,
                batch_rows=max(1, self.batch_rows // len(self.paths)),
                date_col=self.date_col,
                symbols=self.symbols,
                start=self.start,
                end=self.end,
            )
            for path in self.paths
        ]
        iterators = [iter(feed) for feed in feeds]
        pending = []
        timestamp = None
        try:
            for group in heapq.merge(*iterators, key=lambda group: group.timestamp):
                if timestamp is not None and timestamp != group.timestamp:
                    yield self._finish(timestamp, pending)
                    pending = []
                timestamp = group.timestamp
                pending.extend(group.bars)
            if timestamp is not None:
                yield self._finish(timestamp, pending)
        finally:
            for iterator in iterators:
                iterator.close()
            for key in self.scan_stats:
                self.scan_stats[key] = sum(feed.scan_stats.get(key, 0) for feed in feeds)

    @staticmethod
    def _finish(timestamp: datetime, bars: list[Bar]) -> MarketSlice:
        bars.sort(key=lambda bar: bar.symbol)
        if any(bars[i - 1].symbol == bars[i].symbol for i in range(1, len(bars))):
            raise ValueError(f"Duplicate symbol at Parquet timestamp {timestamp}")
        return MarketSlice(timestamp, tuple(bars))
