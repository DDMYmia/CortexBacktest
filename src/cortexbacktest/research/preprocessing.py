"""Cross-sectional factor preprocessing utilities.

Factor matrices are wide (``date`` plus one column per symbol), but every
transform here is cross-sectional: it works within a date. Wide layouts make
that awkward, so each transform unpivots to long form, expresses the operation
as a window over ``date``, and pivots back. That keeps the work inside Polars
rather than looping over rows in Python.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import polars as pl

from .frames import _to_polars_matrix

GroupMap = Mapping[str, str]

_VALUE = "value"
_SYMBOL = "symbol"
_GROUP = "group"


@dataclass(frozen=True, slots=True)
class FactorPreprocessConfig:
    """Configuration for date-wise factor preprocessing.

    The order is fixed: winsorize, group-neutralize, rank, z-score, demean.
    Keeping this as a value object gives the research workflow one stable
    extension point as more factor transforms are added.
    """

    winsorize_quantiles: tuple[float, float] | None = None
    neutralize_groups: GroupMap | None = None
    rank: bool = False
    zscore: bool = False
    demean: bool = False


def winsorize_cross_section(
    factors: pl.DataFrame,
    *,
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
) -> pl.DataFrame:
    """Clip each date's factor values to cross-sectional quantile bounds."""
    _validate_quantiles(lower_quantile, upper_quantile)
    long, symbols = _to_long(factors)
    return _to_wide(long.with_columns(_winsorize_expr(lower_quantile, upper_quantile)), symbols)


def demean_cross_section(factors: pl.DataFrame) -> pl.DataFrame:
    """Subtract each date's cross-sectional mean from non-null factor values."""
    long, symbols = _to_long(factors)
    return _to_wide(long.with_columns(_demean_expr()), symbols)


def zscore_cross_section(factors: pl.DataFrame, *, ddof: int = 0) -> pl.DataFrame:
    """Convert each date's factor values to cross-sectional z-scores."""
    if ddof < 0:
        raise ValueError("ddof must be >= 0")
    long, symbols = _to_long(factors)
    return _to_wide(long.with_columns(_zscore_expr(ddof)), symbols)


def rank_cross_section(factors: pl.DataFrame, *, pct: bool = True) -> pl.DataFrame:
    """Rank factor values within each date's cross-section."""
    long, symbols = _to_long(factors)
    return _to_wide(long.with_columns(_rank_expr(pct=pct)), symbols)


def neutralize_by_group(
    factors: pl.DataFrame,
    groups: GroupMap,
    *,
    missing: str = "keep",
) -> pl.DataFrame:
    """Subtract date-wise group means from each symbol's factor values.

    ``groups`` maps symbols to group labels, typically sector or industry.
    Symbols without a group are preserved by default so incomplete metadata does
    not silently remove tradable names during early research.
    """
    if missing not in {"keep", "nan"}:
        raise ValueError("missing must be 'keep' or 'nan'")
    long, symbols = _to_long(factors)
    return _to_wide(_neutralize(long, groups, missing=missing), symbols)


def prepare_factor(
    factors: pl.DataFrame,
    config: FactorPreprocessConfig | None = None,
    *,
    winsorize_quantiles: tuple[float, float] | None = None,
    neutralize_groups: GroupMap | None = None,
    rank: bool = False,
    zscore: bool = False,
    demean: bool = False,
) -> pl.DataFrame:
    """Apply the standard cross-sectional preprocessing pipeline.

    The whole pipeline runs on one long-form frame, so the matrix is reshaped
    once rather than once per enabled step.
    """
    if config is not None:
        winsorize_quantiles = config.winsorize_quantiles
        neutralize_groups = config.neutralize_groups
        rank = config.rank
        zscore = config.zscore
        demean = config.demean
    if winsorize_quantiles is not None:
        _validate_quantiles(*winsorize_quantiles)

    long, symbols = _to_long(factors)
    if winsorize_quantiles is not None:
        long = long.with_columns(_winsorize_expr(*winsorize_quantiles))
    if neutralize_groups is not None:
        long = _neutralize(long, neutralize_groups, missing="keep")
    if rank:
        long = long.with_columns(_rank_expr(pct=True))
    if zscore:
        long = long.with_columns(_zscore_expr(0))
    if demean:
        long = long.with_columns(_demean_expr())
    return _to_wide(long, symbols)


def _winsorize_expr(lower_quantile: float, upper_quantile: float) -> pl.Expr:
    # Linear interpolation: Polars defaults to "nearest", which would silently
    # disagree with the usual quantile convention.
    return (
        pl.col(_VALUE)
        .clip(
            pl.col(_VALUE).quantile(lower_quantile, interpolation="linear").over("date"),
            pl.col(_VALUE).quantile(upper_quantile, interpolation="linear").over("date"),
        )
        .alias(_VALUE)
    )


def _demean_expr() -> pl.Expr:
    return (pl.col(_VALUE) - pl.col(_VALUE).mean().over("date")).alias(_VALUE)


def _zscore_expr(ddof: int) -> pl.Expr:
    std = pl.col(_VALUE).std(ddof=ddof).over("date")
    mean = pl.col(_VALUE).mean().over("date")
    return (
        pl.when(pl.col(_VALUE).is_null() | std.is_null() | ~std.is_finite())
        .then(None)
        # A flat cross-section carries no dispersion; every observation sits at
        # the mean, so the z-score is 0 rather than a divide-by-zero.
        .when(std == 0)
        .then(pl.lit(0.0))
        .otherwise((pl.col(_VALUE) - mean) / std)
        .alias(_VALUE)
    )


def _rank_expr(*, pct: bool) -> pl.Expr:
    ranks = pl.col(_VALUE).rank(method="average").over("date")
    if pct:
        ranks = ranks / pl.col(_VALUE).count().over("date")
    return ranks.alias(_VALUE)


def _neutralize(long: pl.DataFrame, groups: GroupMap, *, missing: str) -> pl.DataFrame:
    group_map = {str(symbol).upper(): str(group) for symbol, group in groups.items()}
    tagged = long.with_columns(
        pl.col(_SYMBOL).str.to_uppercase().replace_strict(group_map, default=None).alias(_GROUP)
    )
    group_mean = pl.col(_VALUE).mean().over(["date", _GROUP])
    ungrouped = pl.lit(None) if missing == "nan" else pl.col(_VALUE)
    return tagged.with_columns(
        pl.when(pl.col(_GROUP).is_null())
        .then(ungrouped)
        .otherwise(pl.col(_VALUE) - group_mean)
        .alias(_VALUE)
    ).drop(_GROUP)


def _to_long(factors: pl.DataFrame) -> tuple[pl.DataFrame, list[str]]:
    """Unpivot a wide factor matrix, treating non-finite values as missing."""
    data = _to_polars_matrix(factors).sort("date")
    symbols = [column for column in data.columns if column != "date"]
    # With no symbol columns, unpivot would discard the entire date axis.
    long = data.unpivot(index="date", variable_name=_SYMBOL, value_name=_VALUE) if symbols else data.with_columns(
        pl.lit(None, dtype=pl.String).alias(_SYMBOL), pl.lit(None, dtype=pl.Float64).alias(_VALUE))
    return (
        long.with_columns(
            pl.when(pl.col(_VALUE).is_finite())
            .then(pl.col(_VALUE))
            .otherwise(None)
            .cast(pl.Float64)
            .alias(_VALUE)
        ),
        symbols,
    )


def _to_wide(long: pl.DataFrame, symbols: list[str]) -> pl.DataFrame:
    if not symbols:
        return long.select("date").unique().sort("date")
    if long.is_empty():
        return pl.DataFrame(schema={"date": long.schema["date"], **dict.fromkeys(symbols, pl.Float64)})
    return long.pivot(on=_SYMBOL, index="date", values=_VALUE).select(["date", *symbols]).sort("date")


def _validate_quantiles(lower_quantile: float, upper_quantile: float) -> None:
    if not 0 <= lower_quantile <= 1:
        raise ValueError("lower_quantile must be in [0, 1]")
    if not 0 <= upper_quantile <= 1:
        raise ValueError("upper_quantile must be in [0, 1]")
    if lower_quantile > upper_quantile:
        raise ValueError("lower_quantile must be <= upper_quantile")
