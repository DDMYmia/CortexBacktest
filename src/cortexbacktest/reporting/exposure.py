"""Position, exposure, and turnover reporting helpers."""

from __future__ import annotations

from collections.abc import Mapping

import polars as pl

from ..contracts import Position
from ..data.backend import to_backend
from .records import positions_frame


def exposure_frame(equity_curve) -> pl.DataFrame:
    """Extract portfolio exposure columns as a Polars DataFrame."""
    columns = [
        "timestamp",
        "equity",
        "cash",
        "long_exposure",
        "short_exposure",
        "gross_exposure",
        "net_exposure",
        "leverage",
    ]
    data = to_backend(equity_curve, backend="polars")
    available = [column for column in columns if column in data.columns]
    if not available:
        return _empty_polars(columns)
    return data.select(available)


def exposure_summary(equity_curve) -> dict[str, float]:
    """Summarize leverage and exposure ranges from an equity curve using Polars."""
    frame = exposure_frame(equity_curve)
    return {
        "max_gross_exposure": _max_abs_or_zero_pl(frame, "gross_exposure"),
        "max_net_exposure": _column_stat_or_zero(frame, "net_exposure", "max"),
        "min_net_exposure": _column_stat_or_zero(frame, "net_exposure", "min"),
        "max_leverage": _column_stat_or_zero(frame, "leverage", "max"),
        "avg_leverage": _column_stat_or_zero(frame, "leverage", "mean"),
    }


def group_exposure_frame(
    positions: list[Position] | Mapping[str, Position],
    groups: Mapping[str, str],
    *,
    equity: float | None = None,
    unknown_group: str = "unknown",
) -> pl.DataFrame:
    """Aggregate current market value by group as a Polars DataFrame."""
    items = list(positions.values()) if isinstance(positions, Mapping) else list(positions)
    position_rows = positions_frame(items)
    if position_rows.is_empty():
        return _empty_polars(["group", "market_value", "gross_exposure", "weight"])
    group_map = {str(symbol).upper(): str(group) for symbol, group in dict(groups).items()}
    grouped = (
        position_rows.with_columns(
            pl.col("symbol").map_elements(
                lambda symbol: group_map.get(str(symbol).upper(), unknown_group),
                return_dtype=pl.Utf8,
            ).alias("group")
        )
        .group_by("group")
        .agg(
            pl.col("market_value").sum(),
            pl.col("market_value").abs().sum().alias("gross_exposure"),
        )
        .sort("group")
    )
    denominator = equity if equity is not None else grouped["gross_exposure"].sum()
    return grouped.with_columns(
        (pl.col("market_value") / denominator).alias("weight") if denominator else pl.lit(0.0).alias("weight")
    )


def turnover_frame(fills, equity_curve) -> pl.DataFrame:
    """Compute timestamp-level one-way turnover as a Polars DataFrame."""
    fill_frame = to_backend(fills, backend="polars")
    if fill_frame.is_empty():
        return _empty_polars(["timestamp", "traded_notional", "equity", "turnover"])
    required = {"timestamp", "notional"}
    missing = required.difference(fill_frame.columns)
    if missing:
        raise ValueError(f"Missing required fill columns: {sorted(missing)}")
    fill_frame = fill_frame.with_columns(_timestamp_expression(pl.col("timestamp")).alias("timestamp"))
    traded = (
        fill_frame.with_columns(pl.col("notional").abs().alias("traded_notional"))
        .group_by("timestamp")
        .agg(pl.col("traded_notional").sum())
        .sort("timestamp")
    )
    equity_frame = to_backend(equity_curve, backend="polars")
    if equity_frame.is_empty() or "timestamp" not in equity_frame.columns or "equity" not in equity_frame.columns:
        return traded.with_columns([pl.lit(None).alias("equity"), pl.lit(None).alias("turnover")])
    equity = (
        equity_frame.select(["timestamp", "equity"])
        .with_columns(_timestamp_expression(pl.col("timestamp")).alias("timestamp"))
        .unique("timestamp", keep="last")
    )
    return traded.join(equity, on="timestamp", how="left").with_columns(
        (pl.col("traded_notional") / pl.col("equity")).alias("turnover")
    )


def turnover_summary(fills, equity_curve) -> dict[str, float]:
    """Summarize one-way turnover using Polars inputs."""
    input_frame = to_backend(fills, backend="polars")
    frame = input_frame if {"traded_notional", "turnover"}.issubset(input_frame.columns) else turnover_frame(input_frame, equity_curve)
    if frame.is_empty():
        return {"total_traded_notional": 0.0, "total_turnover": 0.0, "avg_turnover": 0.0}
    turnover = frame.select(pl.col("turnover")).drop_nulls()
    return {
        "total_traded_notional": _column_stat_or_zero(frame, "traded_notional", "sum"),
        "total_turnover": _column_stat_or_zero(turnover, "turnover", "sum"),
        "avg_turnover": _column_stat_or_zero(turnover, "turnover", "mean"),
    }


def _max_abs_or_zero_pl(frame: pl.DataFrame, column: str) -> float:
    if frame.is_empty() or column not in frame.columns:
        return 0.0
    value = frame.select(pl.col(column).abs().max()).item()
    return 0.0 if value is None else float(value)


def _column_stat_or_zero(frame: pl.DataFrame, column: str, stat: str) -> float:
    if frame.is_empty() or column not in frame.columns:
        return 0.0
    if stat == "max":
        value = frame.select(pl.col(column).max()).item()
    elif stat == "min":
        value = frame.select(pl.col(column).min()).item()
    elif stat == "mean":
        value = frame.select(pl.col(column).mean()).item()
    elif stat == "sum":
        value = frame.select(pl.col(column).sum()).item()
    else:
        raise ValueError(f"Unsupported column statistic: {stat}")
    return 0.0 if value is None else float(value)


def _empty_polars(columns: list[str]) -> pl.DataFrame:
    return pl.DataFrame({column: [] for column in columns})


def _timestamp_expression(expr):
    return expr.cast(pl.Datetime("us", "UTC"), strict=False)
