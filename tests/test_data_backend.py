import pandas as pd
import polars as pl

from cortexbacktest.data.backend import normalize_date_column
from cortexbacktest.data import (
    read_table,
    scan_table,
    to_backend,
    to_pandas,
    write_table,
)


def test_read_table_defaults_to_polars(tmp_path):
    path = tmp_path / "sample.csv"
    pd.DataFrame({"a": [1, 2], "b": [3, 4]}).to_csv(path, index=False)

    frame = read_table(path)

    assert isinstance(frame, pl.DataFrame)
    assert frame["a"].to_list() == [1, 2]


def test_normalized_datetime_keeps_precision_timezone_and_independent_columns():
    frame = pl.DataFrame({"date": [1_700_000_000_000_000_123], "value": [1]}).with_columns(
        pl.col("date").cast(pl.Datetime("ns", "Pacific/Auckland")))
    normalized = normalize_date_column(frame)
    assert normalized.equals(frame) and normalized.schema == frame.schema
    normalized.drop_in_place("value")
    assert frame.columns == ["date", "value"]


def test_to_backend_round_trips_pandas():
    frame = pd.DataFrame({"a": [1, 2]})

    converted = to_backend(frame, backend="pandas")

    assert isinstance(converted, pd.DataFrame)
    assert to_pandas(converted).equals(frame)


def test_write_table_round_trips_csv(tmp_path):
    path = tmp_path / "sample.csv"
    source = pd.DataFrame({"a": [1, 2], "b": [3.0, 4.0]})

    written = write_table(source, path)
    loaded = read_table(written)

    assert written == path
    assert loaded["a"].to_list() == [1, 2]
    assert loaded["b"].to_list() == [3.0, 4.0]


def test_write_table_parquet_round_trip(tmp_path):
    path = tmp_path / "sample.parquet"
    source = pd.DataFrame({"a": [1, 2]})

    written = write_table(source, path)

    loaded = read_table(written)
    assert loaded["a"].to_list() == [1, 2]


def test_polars_backend_reads_and_scans(tmp_path):
    path = tmp_path / "sample.csv"
    pd.DataFrame({"a": [1, 2]}).to_csv(path, index=False)

    polars_frame = read_table(path, backend="polars")
    assert to_pandas(polars_frame)["a"].tolist() == [1, 2]
    lazy = scan_table(path)
    assert to_pandas(lazy)["a"].tolist() == [1, 2]


def test_polars_lazy_conversion_never_round_trips_through_pandas(monkeypatch):
    from cortexbacktest.data import backend

    def unexpected_pandas(frame):
        raise AssertionError("Polars input must not require pandas")

    monkeypatch.setattr(backend, "to_pandas", unexpected_pandas)
    frame = pl.DataFrame({"a": pl.Series([1, None], dtype=pl.Int32)})
    assert to_backend(frame.lazy(), backend="polars").equals(frame)


def test_default_conversion_and_explicit_pandas_boundary(tmp_path):
    source = pd.DataFrame({"value": [1.0, None]})
    converted = to_backend(source)
    assert isinstance(converted, pl.DataFrame)
    assert converted["value"].to_list() == [1.0, None]
    path = write_table(converted, tmp_path / "data.parquet")
    assert isinstance(read_table(path), pl.DataFrame)
    external = read_table(path, backend="pandas")
    assert isinstance(external, pd.DataFrame)
    assert external.equals(source)


def test_explicit_index_export_preserves_named_pandas_index(tmp_path):
    source = pd.DataFrame({"value": [3.0, 4.0]}, index=pd.Index([10, 11], name="source_id"))
    path = write_table(source, tmp_path / "indexed.csv", index=True)
    assert read_table(path).to_dict(as_series=False) == {"source_id": [10, 11], "value": [3.0, 4.0]}
