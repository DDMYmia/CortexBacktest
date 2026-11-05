"""Polars data I/O with explicit pandas conversion at compatibility boundaries."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal, overload

import polars as pl

if TYPE_CHECKING:
    import pandas as pd

DataFrameBackend = Literal["pandas", "polars"]


@overload
def read_table(path: str | Path, *, backend: Literal["polars"] = ...) -> pl.DataFrame: ...
@overload
def read_table(path: str | Path, *, backend: Literal["pandas"]) -> pd.DataFrame: ...
def read_table(path: str | Path, *, backend: DataFrameBackend = "polars"):
    """Read CSV/parquet as pandas or Polars DataFrame."""
    file_path = Path(path)
    suffix = file_path.suffix.lower()
    if backend == "pandas":
        import pandas as reader
    else:
        reader = pl

    if suffix == ".csv":
        return reader.read_csv(file_path)
    if suffix in {".parquet", ".pq"}:
        return reader.read_parquet(file_path)
    raise ValueError(f"Unsupported file type: {file_path.suffix}")


def write_table(frame, path: str | Path, *, backend: DataFrameBackend = "polars", index: bool = False) -> Path:
    """Write pandas/Polars-compatible DataFrame to CSV/parquet."""
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = file_path.suffix.lower()
    if backend == "pandas" or index:
        # Explicit index export is a pandas boundary; Polars has no implicit row index.
        data = to_pandas(frame)
        if suffix == ".csv":
            data.to_csv(file_path, index=index)
            return file_path
        if suffix in {".parquet", ".pq"}:
            data.to_parquet(file_path, index=index)
            return file_path
        raise ValueError(f"Unsupported file type: {file_path.suffix}")

    data = to_backend(frame, backend="polars")
    if suffix == ".csv":
        data.write_csv(file_path)
        return file_path
    if suffix in {".parquet", ".pq"}:
        data.write_parquet(file_path)
        return file_path
    raise ValueError(f"Unsupported file type: {file_path.suffix}")


def scan_table(path: str | Path):
    """Create a Polars lazy scan for CSV/parquet files."""
    file_path = Path(path)
    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        return pl.scan_csv(file_path)
    if suffix in {".parquet", ".pq"}:
        return pl.scan_parquet(file_path)
    raise ValueError(f"Unsupported file type: {file_path.suffix}")


def to_pandas(frame) -> pd.DataFrame:
    """Convert supported DataFrame objects to pandas."""
    import pandas as pd

    if isinstance(frame, pd.DataFrame):
        return frame
    if isinstance(frame, pl.DataFrame):
        return frame.to_pandas()
    if isinstance(frame, pl.LazyFrame):
        return frame.collect().to_pandas()
    raise TypeError(f"Unsupported frame type: {type(frame)!r}")


@overload
def to_backend(frame, *, backend: Literal["polars"] = ...) -> pl.DataFrame: ...
@overload
def to_backend(frame, *, backend: Literal["pandas"]) -> pd.DataFrame: ...
def to_backend(frame, *, backend: DataFrameBackend = "polars"):
    """Convert a DataFrame to the requested backend."""
    if backend == "pandas":
        return to_pandas(frame)
    if isinstance(frame, pl.DataFrame):
        return frame
    if isinstance(frame, pl.LazyFrame):
        return frame.collect()
    return pl.from_pandas(to_pandas(frame))


def normalize_date_column(frame: pl.DataFrame, column: str = "date") -> pl.DataFrame:
    """Coerce a date column to Datetime, branching on its dtype.

    Sources deliver dates as ISO strings (CSV), Date, or Datetime. Polars
    evaluates both arms of a ``when/then/otherwise``, so a value-pattern branch
    would still run a String->Datetime cast -- which Polars 2.0 removes. Reading
    the schema keeps exactly one conversion on the table.
    """
    if column not in frame.columns:
        raise ValueError(f"frame requires a {column!r} column")
    dtype = frame.schema[column]
    if dtype == pl.String:
        expr = pl.col(column).str.to_datetime(strict=False)
    elif isinstance(dtype, pl.Datetime):
        return frame.clone()
    else:
        expr = pl.col(column).cast(pl.Datetime, strict=False)
    return frame.with_columns(expr.alias(column))
