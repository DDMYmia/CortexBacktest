"""Simple target-weight portfolio simulation."""

from __future__ import annotations

import polars as pl

from .frames import _to_polars_matrix


def simulate_target_weights(
    returns: pl.DataFrame,
    target_weights: pl.DataFrame,
    *,
    transaction_cost_bps: float = 0.0,
    initial_equity: float = 1.0,
) -> pl.DataFrame:
    """Simulate lagged target weights; an unvalued holding makes NAV unknown.

    Missing returns for zero holdings are harmless. Once a held asset cannot be
    valued, later observations cannot repair the missing wealth increment.
    """
    if initial_equity <= 0:
        raise ValueError("initial_equity must be positive")
    if transaction_cost_bps < 0:
        raise ValueError("transaction_cost_bps must be non-negative")

    returns_frame = _to_polars_matrix(returns).sort("date")
    weights_frame = _to_polars_matrix(target_weights).sort("date")
    symbols = sorted(set(weights_frame.columns) - {"date"})
    if not symbols:
        return pl.DataFrame({"date": returns_frame["date"], "equity": [initial_equity] * returns_frame.height})

    joined = returns_frame.select("date", *[
        pl.col(symbol) if symbol in returns_frame.columns else pl.lit(None, dtype=pl.Float64).alias(symbol)
        for symbol in symbols
    ]).join(
        weights_frame.select(["date", *symbols]),
        on="date",
        how="left",
        suffix="_weight",
    )
    shifted_weight_exprs = [
        pl.col(f"{symbol}_weight").shift(1).fill_null(0.0).alias(f"{symbol}_shifted_weight")
        for symbol in symbols
    ]
    weighted_return_exprs = [
        pl.when(pl.col(f"{symbol}_shifted_weight") != 0)
        .then(pl.col(f"{symbol}_shifted_weight") * pl.col(symbol))
        .otherwise(0.0)
        for symbol in symbols
    ]
    unavailable_exprs = [
        (pl.col(f"{symbol}_shifted_weight") != 0) & ~pl.col(symbol).is_finite().fill_null(False)
        for symbol in symbols
    ]
    turnover_exprs = [
        pl.col(f"{symbol}_shifted_weight").diff().abs().fill_null(pl.col(f"{symbol}_shifted_weight").abs())
        for symbol in symbols
    ]
    return (
        joined.with_columns(shifted_weight_exprs)
        .with_columns(
            [
                pl.sum_horizontal(weighted_return_exprs).alias("gross_return"),
                pl.sum_horizontal(turnover_exprs).alias("turnover"),
                pl.any_horizontal(unavailable_exprs).cum_max().alias("unvalued"),
            ]
        )
        .with_columns((pl.col("gross_return") - pl.col("turnover") * (transaction_cost_bps / 10_000.0)).alias("net_return"))
        .with_columns(pl.when(~pl.col("unvalued"))
                      .then(initial_equity * (1.0 + pl.col("net_return")).cum_prod())
                      .otherwise(None).alias("equity"))
    ).select(["date", "equity"])
