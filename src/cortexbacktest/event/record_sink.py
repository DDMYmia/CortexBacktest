"""Streaming event record sinks."""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import BinaryIO, Literal, TextIO

import orjson
import polars as pl
import pyarrow.parquet as pq

from ..contracts import Fill, Order, OrderRecord, Position
from ..record_rows import (
    FILL_COLUMNS,
    OPEN_ORDER_COLUMNS,
    ORDER_COLUMNS,
    POSITION_COLUMNS,
    _scalar_value,
    iter_fill_rows,
    iter_open_order_rows,
    iter_order_record_rows,
    iter_position_rows,
)

RecordFormat = Literal["csv", "ndjson", "parquet"]

#: Rows buffered per writer before an automatic flush. 1,000-10,000 is the
#: range where batching pays for itself without holding an unbounded amount of
#: a long run's orders/fills in memory between flushes.
DEFAULT_BUFFER_ROWS = 2_000

# Stable scalar export contract: a first batch of null reasons/prices must
# not fix the field to Arrow null, nor may integer quantities truncate later
# partial-fill fractions. Timestamps are ISO strings through _scalar_value.
_NUMERIC_RECORD_COLUMNS = {
    "quantity", "price", "commission", "slippage", "notional", "last_price",
    "market_value", "limit_price", "stop_price", "trail_amount", "trail_percent",
}


class EventRecordSink:
    """Write event records as they are produced by the broker.

    Records are buffered (`buffer_rows` per writer, default
    `DEFAULT_BUFFER_ROWS`) and written in batches rather than one syscall and
    one row-encode per record — the shape a long high-turnover run or an order
    audit trail actually writes in. Call `flush()` to force a batch out before
    the buffer fills (e.g. for a durability checkpoint); `close()` (or exiting
    the `with` block) always flushes what remains.

    `fsync=True` additionally asks the OS to commit each flushed batch to disk
    (`os.fsync`) before returning, for a run where surviving a crash matters
    more than flush latency (csv/ndjson only; Parquet ignores `fsync` and is
    finalized on close, without a crash-durability guarantee).
    """

    def __init__(
        self,
        path: str | Path,
        *,
        format: RecordFormat = "csv",
        buffer_rows: int = DEFAULT_BUFFER_ROWS,
        fsync: bool = False,
    ) -> None:
        if format not in {"csv", "ndjson", "parquet"}:
            raise ValueError("format must be 'csv', 'ndjson', or 'parquet'")
        if buffer_rows < 1:
            raise ValueError("buffer_rows must be at least 1")
        self.path = Path(path)
        self.format: RecordFormat = format
        self.buffer_rows = buffer_rows
        self.fsync = fsync
        self.path.mkdir(parents=True, exist_ok=True)
        self._order_writer = _RowWriter(self.path, "orders", ORDER_COLUMNS, format=format, buffer_rows=buffer_rows, fsync=fsync)
        self._fill_writer = _RowWriter(self.path, "fills", FILL_COLUMNS, format=format, buffer_rows=buffer_rows, fsync=fsync)
        self._closed = False

    def write_order_record(self, record: OrderRecord) -> None:
        self._order_writer.write(next(iter_order_record_rows((record,))))

    def write_fill(self, fill: Fill) -> None:
        self._fill_writer.write(next(iter_fill_rows((fill,))))

    def flush(self) -> None:
        """Force any buffered rows out now, without closing the sink."""
        self._order_writer.flush()
        self._fill_writer.flush()

    def write_final_tables(
        self,
        *,
        equity_curve: pl.DataFrame,
        open_orders: list[Order],
        positions: list[Position],
    ) -> Path:
        if self.format == "parquet":
            equity_curve.write_parquet(self.path / "equity_curve.parquet")
        elif self.format == "csv":
            equity_curve.write_csv(self.path / "equity_curve.csv")
        else:
            equity_curve.write_ndjson(self.path / "equity_curve.ndjson")
        _write_rows(self.path, "open_orders", OPEN_ORDER_COLUMNS, iter_open_order_rows(open_orders), format=self.format)
        _write_rows(self.path, "positions", POSITION_COLUMNS, iter_position_rows(positions), format=self.format)
        return self.path

    def close(self) -> None:
        if self._closed:
            return
        self._order_writer.close()
        self._fill_writer.close()
        self._closed = True

    def __enter__(self) -> "EventRecordSink":
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.close()


class _RowWriter:
    """Buffers rows and flushes them as one batch per format's fast path.

    - **csv**: `csv.DictWriter.writerows` once per flush instead of
      `writerow` once per record.
    - **ndjson**: `orjson` (already a core dependency, used elsewhere in this
      package) encodes the whole batch and writes it as one `bytes` blob,
      rather than `json.dumps` plus two `str` writes per record.
    - **parquet**: batches are converted to an Arrow table (via Polars, for a
      column-oriented build rather than one dict per row) and appended
      through a single `pyarrow.parquet.ParquetWriter` for the file's
      lifetime — the writer never re-reads or rewrites rows already on disk.
      All batches use the declared record-column types, including empty files
      and initially all-null optional fields.
    """

    def __init__(
        self, directory: Path, stem: str, columns: list[str], *, format: RecordFormat, buffer_rows: int, fsync: bool = False
    ) -> None:
        self.columns = columns
        self._schema = {column: pl.Float64 if column in _NUMERIC_RECORD_COLUMNS else pl.String
                        for column in columns}
        self.format: RecordFormat = format
        self.buffer_rows = buffer_rows
        self.fsync = fsync
        self._buffer: list[dict] = []
        self._parquet_writer: pq.ParquetWriter | None = None
        self._text_handle: TextIO | None = None
        self._binary_handle: BinaryIO | None = None
        self._csv_writer: csv.DictWriter | None = None
        directory.mkdir(parents=True, exist_ok=True)

        self.path = directory / f"{stem}.{format}"
        if format == "csv":
            self._text_handle = self.path.open("w", newline="", encoding="utf-8")
            self._csv_writer = csv.DictWriter(self._text_handle, fieldnames=columns, extrasaction="ignore")
            self._csv_writer.writeheader()
        elif format == "ndjson":
            self._binary_handle = self.path.open("wb")

    def write(self, row: dict) -> None:
        self._buffer.append(row)
        if len(self._buffer) >= self.buffer_rows:
            self.flush()

    def flush(self) -> None:
        if not self._buffer:
            return
        batch, self._buffer = self._buffer, []
        if self.format == "csv":
            assert self._csv_writer is not None
            assert self._text_handle is not None
            self._csv_writer.writerows({column: _scalar_value(row.get(column)) for column in self.columns} for row in batch)
            self._text_handle.flush()
            if self.fsync:
                os.fsync(self._text_handle.fileno())
        elif self.format == "ndjson":
            assert self._binary_handle is not None
            encoded = b"\n".join(
                orjson.dumps({key: _scalar_value(value) for key, value in row.items()}, option=orjson.OPT_SORT_KEYS)
                for row in batch
            )
            self._binary_handle.write(encoded)
            self._binary_handle.write(b"\n")
            self._binary_handle.flush()
            if self.fsync:
                os.fsync(self._binary_handle.fileno())
        else:
            table = pl.DataFrame(
                {column: [_scalar_value(row.get(column)) for row in batch] for column in self.columns},
                schema=self._schema,
            ).to_arrow()
            if self._parquet_writer is None:
                self._parquet_writer = pq.ParquetWriter(self.path, table.schema)
            self._parquet_writer.write_table(table)

    def close(self) -> None:
        self.flush()
        if self._text_handle is not None:
            self._text_handle.close()
        if self._binary_handle is not None:
            self._binary_handle.close()
        if self._parquet_writer is not None:
            self._parquet_writer.close()
        elif self.format == "parquet":
            # Nothing was ever written (e.g. a run with zero fills): leave an
            # empty-but-valid Parquet file rather than none at all, so a
            # downstream `pl.read_parquet` on every run's output does not need
            # a special case for "no rows happened".
            pl.DataFrame(schema=self._schema).write_parquet(self.path)


def _write_rows(directory: Path, stem: str, columns: list[str], rows, *, format: RecordFormat) -> None:
    writer = _RowWriter(directory, stem, columns, format=format, buffer_rows=DEFAULT_BUFFER_ROWS)
    try:
        for row in rows:
            writer.write(row)
    finally:
        writer.close()
