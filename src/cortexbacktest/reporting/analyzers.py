"""Analysis helpers for event-driven backtest results."""

from __future__ import annotations

import polars as pl


def trade_summary(fills: pl.DataFrame) -> dict[str, float | int]:
    """Summarize fill records without converting through pandas."""
    if fills.is_empty():
        return {
            "fill_count": 0,
            "buy_count": 0,
            "sell_count": 0,
            "gross_notional": 0.0,
            "commission": 0.0,
            "slippage": 0.0,
        }
    required = {"side", "notional", "commission", "slippage"}
    missing = required.difference(fills.columns)
    if missing:
        raise ValueError(f"fill records missing columns: {sorted(missing)}")
    summary = fills.select(
        [
            pl.len().alias("fill_count"),
            (pl.col("side") == "buy").sum().alias("buy_count"),
            (pl.col("side") == "sell").sum().alias("sell_count"),
            pl.col("notional").abs().sum().alias("gross_notional"),
            pl.col("commission").sum().alias("commission"),
            pl.col("slippage").sum().alias("slippage"),
        ]
    ).row(0, named=True)
    return {
        "fill_count": int(summary["fill_count"]),
        "buy_count": int(summary["buy_count"]),
        "sell_count": int(summary["sell_count"]),
        "gross_notional": float(summary["gross_notional"] or 0.0),
        "commission": float(summary["commission"] or 0.0),
        "slippage": float(summary["slippage"] or 0.0),
    }
