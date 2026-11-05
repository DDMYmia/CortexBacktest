"""Record table helpers for orders, fills, and equity curves."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

import polars as pl

from ..contracts import Fill, Order, OrderRecord, Position

from ..record_rows import (
    ORDER_COLUMNS,
    OPEN_ORDER_COLUMNS,
    FILL_COLUMNS,
    POSITION_COLUMNS,
    iter_order_record_rows,
    iter_open_order_rows,
    iter_fill_rows,
    iter_position_rows,
    _scalar_value,
)


def order_records_frame(records: list[OrderRecord]) -> pl.DataFrame:
    return _polars_frame(iter_order_record_rows(records), ORDER_COLUMNS, floats=("quantity",))


def open_orders_frame(orders: list[Order]) -> pl.DataFrame:
    return _polars_frame(iter_open_order_rows(orders), OPEN_ORDER_COLUMNS, floats=("quantity",))


def fills_frame(fills: list[Fill]) -> pl.DataFrame:
    return _polars_frame(
        iter_fill_rows(fills), FILL_COLUMNS,
        floats=("quantity", "price", "commission", "slippage", "notional"),
    )


def positions_frame(positions: list[Position]) -> pl.DataFrame:
    return _polars_frame(
        iter_position_rows(positions), POSITION_COLUMNS,
        floats=("quantity", "last_price", "market_value"),
    ).sort("symbol")


def write_order_records_csv(records: Iterable[OrderRecord], path: str | Path) -> Path:
    return _write_csv(iter_order_record_rows(records), ORDER_COLUMNS, path)


def write_order_records_ndjson(records: Iterable[OrderRecord], path: str | Path) -> Path:
    return _write_ndjson(iter_order_record_rows(records), path)


def write_fills_csv(fills: Iterable[Fill], path: str | Path) -> Path:
    return _write_csv(iter_fill_rows(fills), FILL_COLUMNS, path)


def write_fills_ndjson(fills: Iterable[Fill], path: str | Path) -> Path:
    return _write_ndjson(iter_fill_rows(fills), path)


def write_open_orders_csv(orders: Iterable[Order], path: str | Path) -> Path:
    return _write_csv(iter_open_order_rows(orders), OPEN_ORDER_COLUMNS, path)


def write_open_orders_ndjson(orders: Iterable[Order], path: str | Path) -> Path:
    return _write_ndjson(iter_open_order_rows(orders), path)


def write_positions_csv(positions: Iterable[Position], path: str | Path) -> Path:
    return _write_csv(iter_position_rows(positions), POSITION_COLUMNS, path)


def write_positions_ndjson(positions: Iterable[Position], path: str | Path) -> Path:
    return _write_ndjson(iter_position_rows(positions), path)


def _polars_frame(rows: Iterable[dict], columns: list[str], *, floats: tuple[str, ...] = ()) -> pl.DataFrame:
    """Use the streaming export's row contract for in-memory report tables too."""
    records = list(rows)
    if not records:
        return pl.DataFrame({column: [] for column in columns})
    return pl.DataFrame(
        records, infer_schema_length=None,
        schema_overrides={column: pl.Float64 for column in floats},
    ).select(columns)


def _write_csv(rows: Iterable[dict], columns: list[str], path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: _scalar_value(row.get(column)) for column in columns})
    return output_path


def _write_ndjson(rows: Iterable[dict], path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            values = {key: _scalar_value(value) for key, value in row.items()}
            handle.write(json.dumps(values, sort_keys=True))
            handle.write("\n")
    return output_path
