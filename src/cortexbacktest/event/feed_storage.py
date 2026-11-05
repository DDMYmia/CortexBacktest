"""Bounded Arrow IPC transfer for process workers, preserving all Bar fields."""
from __future__ import annotations

from itertools import islice
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.ipc as ipc

from ..contracts import Bar
from .feed import ColumnarBarFeed

BAR_FIELDS = Bar._fields


def write_feed_ipc(feed: ColumnarBarFeed, path: str | Path, *, batch_rows: int = 50_000) -> None:
    if batch_rows < 1:
        raise ValueError("batch_rows must be positive")
    bars = (bar for group in feed for bar in group.bars)
    schema = pa.schema([(name, pa.string() if name == "symbol" else pa.timestamp("us", tz="UTC")
                         if name == "timestamp" else pa.bool_() if name == "halted" else pa.float64())
                        for name in BAR_FIELDS])
    # Arrow timestamps retain awareness. Naive input remains naive after reading.
    chunk = list(islice(bars, batch_rows))
    if chunk and chunk[0].timestamp.tzinfo is None:
        schema = schema.set(schema.get_field_index("timestamp"), pa.field("timestamp", pa.timestamp("us")))
    with pa.OSFile(str(path), "wb") as sink, ipc.new_file(sink, schema) as writer:
        while chunk:
            writer.write_table(pa.table({name: [getattr(bar, name) for bar in chunk] for name in BAR_FIELDS}, schema=schema))
            chunk = list(islice(bars, batch_rows))


def read_feed_ipc(path: str | Path, *, cache: bool = True) -> ColumnarBarFeed:
    return ColumnarBarFeed.from_polars(pl.read_ipc(path, memory_map=True), date_col="timestamp", cache=cache)
