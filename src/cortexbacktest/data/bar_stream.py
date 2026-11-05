"""Replayable, date-partitioned bar feed with memory bounded by one partition."""
from __future__ import annotations

from collections.abc import Callable, Iterator

import polars as pl

from ..event.feed import ColumnarBarFeed, MarketSlice


class PartitionedBarFeed(ColumnarBarFeed):
    """Iterate one sorted date partition at a time; ``bars`` materializes all rows.

    The partition factory must be replayable and globally ordered. Iteration
    validates timestamp boundaries, including duplicate dates across partitions.
    Summary/final-snapshot engine mode also bounds result retention.
    """
    def __init__(self, partitions: Callable[[], Iterator[pl.LazyFrame]]):
        self._partitions = partitions
        self._data_fingerprint = None

    @property
    def single_bars(self):
        return None

    def __iter__(self) -> Iterator[MarketSlice]:
        last = None
        for partition in self._partitions():
            frame = partition.sort(["timestamp", "symbol"]).collect(engine="streaming")
            if frame.is_empty():
                continue
            feed = ColumnarBarFeed.from_polars(frame, date_col="timestamp", cache=False)
            for group in feed:
                if last is not None and group.timestamp <= last:
                    raise ValueError("bar partitions must have strictly increasing timestamp boundaries")
                last = group.timestamp
                yield group

    def __len__(self):
        return sum(1 for _ in self)

    @property
    def bars(self):
        return [bar for group in self for bar in group.bars]
