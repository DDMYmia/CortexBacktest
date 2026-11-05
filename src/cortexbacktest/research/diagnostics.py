"""Polars-native diagnostics for cross-sectional factor research."""

from __future__ import annotations

import math

import polars as pl

from ..data.backend import to_backend


def ic_summary(values, *, value_col: str = "rank_ic") -> pl.DataFrame:
    """Summarize an IC series with stability and significance diagnostics."""
    frame = to_backend(values, backend="polars")
    if value_col not in frame.columns:
        raise ValueError(f"IC data requires a {value_col!r} column")
    clean = frame.select(pl.col(value_col).cast(pl.Float64, strict=False).alias(value_col)).filter(
        pl.col(value_col).is_finite()
    )
    if clean.is_empty():
        return pl.DataFrame(
            {
                "count": [0],
                "mean": [None],
                "std": [None],
                "information_ratio": [None],
                "t_stat": [None],
                "positive_ratio": [None],
            }
        )
    count = clean.height
    mean = clean[value_col].mean()
    std = clean[value_col].std(ddof=1) if count > 1 else None
    information_ratio = None
    t_stat = None
    if std is not None and std != 0 and not math.isnan(std):
        information_ratio = float(mean / std)
        t_stat = float(information_ratio * math.sqrt(count))
    positive_ratio = clean.select((pl.col(value_col) > 0).mean()).item()
    return pl.DataFrame(
        {
            "count": [count],
            "mean": [mean],
            "std": [std],
            "information_ratio": [information_ratio],
            "t_stat": [t_stat],
            "positive_ratio": [positive_ratio],
        }
    )


def cumulative_quantile_returns(quantile_returns) -> pl.DataFrame:
    """Compound each quantile return series into a cumulative return curve."""
    frame = to_backend(quantile_returns, backend="polars")
    if "date" not in frame.columns:
        raise ValueError("quantile returns require a 'date' column")
    value_columns = [column for column in frame.columns if column != "date"]
    if not value_columns:
        return frame.select("date")
    value = pl.col(value_columns)
    return frame.sort("date").select("date",
        pl.when(value.is_finite().fill_null(False).cum_min())
        .then((value + 1.0).cum_prod() - 1.0).otherwise(None))


def factor_rank_autocorrelation(factors, *, periods: int = 1) -> pl.DataFrame:
    """Compute cross-sectional factor-rank autocorrelation through time."""
    if periods < 1:
        raise ValueError("periods must be >= 1")
    frame = to_backend(factors, backend="polars")
    if "date" not in frame.columns:
        raise ValueError("factor matrix requires a 'date' column")
    symbols = [column for column in frame.columns if column != "date"]
    if not symbols or frame.height <= periods:
        return pl.DataFrame({"date": [], "rank_autocorrelation": []})
    long = (
        frame.sort("date")
        .with_row_index("_period")
        .unpivot(index=["_period", "date"], variable_name="symbol", value_name="factor")
        .filter(pl.col("factor").is_finite())
        .with_columns(pl.col("factor").rank("average").over("_period").alias("_rank"))
    )
    previous = long.select(
        [
            (pl.col("_period") + periods).alias("_period"),
            pl.col("symbol"),
            pl.col("_rank").alias("_previous_rank"),
        ]
    )
    return (
        long.join(previous, on=["_period", "symbol"], how="inner")
        .group_by(["_period", "date"])
        .agg(pl.corr("_rank", "_previous_rank").alias("rank_autocorrelation"))
        .drop_nulls("rank_autocorrelation")
        .sort("_period")
        .select(["date", "rank_autocorrelation"])
    )


def factor_information_coefficient(
    frame,
    *,
    method: str = "spearman",
    group_adjust: bool = False,
    by_group: bool = False,
    date_col: str = "date",
    symbol_col: str = "symbol",
    factor_col: str = "factor",
    forward_return_col: str = "forward_return",
    group_col: str = "group",
) -> pl.DataFrame:
    """Compute date-wise factor IC, optionally group-neutral or by group."""
    if method not in {"pearson", "spearman"}:
        raise ValueError("method must be 'pearson' or 'spearman'")
    data = to_backend(frame, backend="polars")
    required = {date_col, symbol_col, factor_col, forward_return_col}
    if group_adjust or by_group:
        required.add(group_col)
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"factor data missing columns: {sorted(missing)}")
    selected = [
        pl.col(date_col).alias("date"),
        pl.col(symbol_col).cast(pl.Utf8).alias("symbol"),
        pl.col(factor_col).cast(pl.Float64, strict=False).alias("factor"),
        pl.col(forward_return_col).cast(pl.Float64, strict=False).alias("forward_return"),
    ]
    if group_adjust or by_group:
        selected.append(pl.col(group_col).cast(pl.Utf8).alias("group"))
    data = data.select(selected).drop_nulls().filter(
        pl.col("factor").is_finite() & pl.col("forward_return").is_finite())
    if data.is_empty():
        columns = ["date", *(["group"] if by_group else []), "rank_ic" if method == "spearman" else "ic"]
        return pl.DataFrame({column: [] for column in columns})
    if group_adjust:
        data = data.with_columns(
            (
                pl.col("forward_return")
                - pl.col("forward_return").mean().over(["date", "group"])
            ).alias("forward_return")
        )
    group_keys = ["date", *(["group"] if by_group else [])]
    if method == "spearman":
        data = data.with_columns(pl.col("factor", "forward_return").rank("average").over(group_keys))
    value_column = "rank_ic" if method == "spearman" else "ic"
    usable = (
        data.group_by(group_keys)
        .agg(
            [
                pl.len().alias("_count"),
                pl.col("factor").n_unique().alias("_factor_unique"),
                pl.col("forward_return").n_unique().alias("_return_unique"),
                pl.corr("factor", "forward_return").alias(value_column),
            ]
        )
        .filter(
            (pl.col("_count") >= 2)
            & (pl.col("_factor_unique") >= 2)
            & (pl.col("_return_unique") >= 2)
        )
        .drop_nulls(value_column)
        .sort(group_keys)
    )
    return usable.select([*group_keys, value_column])


def quantile_turnover(
    factors,
    *,
    quantiles: int = 5,
    periods: int = 1,
) -> pl.DataFrame:
    """Measure membership turnover for each factor quantile."""
    if quantiles < 2:
        raise ValueError("quantiles must be >= 2")
    if periods < 1:
        raise ValueError("periods must be >= 1")
    frame = to_backend(factors, backend="polars")
    if "date" not in frame.columns:
        raise ValueError("factor matrix requires a 'date' column")
    symbols = [column for column in frame.columns if column != "date"]
    if not symbols or frame.height <= periods:
        return pl.DataFrame({"date": [], "quantile": [], "turnover": []})
    assignments = (
        frame.sort("date")
        .with_row_index("_period")
        .unpivot(index=["_period", "date"], variable_name="symbol", value_name="factor")
        .filter(pl.col("factor").is_finite())
        .with_columns(
            [
                pl.col("factor").rank("ordinal").over("_period").alias("_rank"),
                pl.len().over("_period").alias("_count"),
            ]
        )
        .with_columns(
            (
                ((pl.col("_rank") - 1) * quantiles / pl.col("_count")).floor() + 1
            )
            .cast(pl.Int64)
            .alias("quantile")
        )
        .select(["_period", "date", "symbol", "quantile"])
    )
    previous = assignments.select(
        [
            (pl.col("_period") + periods).alias("_period"),
            pl.col("symbol"),
            pl.col("quantile"),
            pl.lit(True).alias("_retained"),
        ]
    )
    return (
        assignments.join(previous, on=["_period", "symbol", "quantile"], how="left")
        .filter(pl.col("_period") >= periods)
        .group_by(["_period", "date", "quantile"])
        .agg(
            (
                1.0 - pl.col("_retained").fill_null(False).cast(pl.Float64).mean()
            ).alias("turnover")
        )
        .sort(["_period", "quantile"])
        .select(["date", "quantile", "turnover"])
    )


def factor_decay_summary(report) -> pl.DataFrame:
    """Summarize Rank IC stability across each forward-return horizon."""
    rows = []
    for horizon, result in sorted(report.horizon_results.items()):
        summary = ic_summary(result.rank_ic_by_date, value_col="rank_ic").row(0, named=True)
        rows.append({"horizon": horizon, **summary})
    if not rows:
        return pl.DataFrame(
            schema={
                "horizon": pl.Int64,
                "count": pl.Int64,
                "mean": pl.Float64,
                "std": pl.Float64,
                "information_ratio": pl.Float64,
                "t_stat": pl.Float64,
                "positive_ratio": pl.Float64,
            }
        )
    return pl.DataFrame(rows).sort("horizon")
