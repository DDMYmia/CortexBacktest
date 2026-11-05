"""Multi-factor construction and time-series factor decay."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping

import polars as pl

from .frames import _to_polars_matrix
from .preprocessing import FactorPreprocessConfig, prepare_factor


@dataclass(frozen=True, slots=True)
class FactorSpec:
    """Named factor input for multi-factor construction."""

    name: str
    values: object
    weight: float = 1.0
    preprocessing: FactorPreprocessConfig | None = None


def align_factor_matrices(
    factors: Mapping[str, object] | list[FactorSpec],
    *,
    join: str = "inner",
) -> dict[str, pl.DataFrame]:
    """Align Polars factor matrices to a common date x symbol shape."""
    specs = _normalize_factor_specs(factors)
    if not specs:
        raise ValueError("factors must not be empty")
    if join not in {"inner", "outer"}:
        raise ValueError("join must be 'inner' or 'outer'")

    matrices = {spec.name: _factor_matrix_pl(spec.values) for spec in specs}
    date_sets = [set(matrix["date"]) for matrix in matrices.values()]
    symbol_sets = [set(_symbols(matrix)) for matrix in matrices.values()]
    dates = sorted(set.intersection(*date_sets) if join == "inner" else set.union(*date_sets))
    symbols = sorted(set.intersection(*symbol_sets) if join == "inner" else set.union(*symbol_sets))

    base = pl.DataFrame({"date": dates}, schema={"date": next(iter(matrices.values())).schema["date"]})
    result = {}
    for spec in specs:
        matrix = matrices[spec.name]
        available = [symbol for symbol in symbols if symbol in matrix.columns]
        aligned = base.join(matrix.select(["date", *available]), on="date", how="left")
        missing = [symbol for symbol in symbols if symbol not in aligned.columns]
        if missing:
            aligned = aligned.with_columns([pl.lit(None, dtype=pl.Float64).alias(symbol) for symbol in missing])
        result[spec.name] = aligned.select(["date", *symbols])
    return result


def combine_factor_scores(
    factors: Mapping[str, object] | list[FactorSpec],
    *,
    weights: Mapping[str, float] | None = None,
    preprocessing: FactorPreprocessConfig | None = None,
    normalize: bool = True,
    min_count: int = 1,
    join: str = "inner",
) -> pl.DataFrame:
    """Combine named Polars factor matrices into one composite score matrix."""
    specs = _normalize_factor_specs(factors, weights=weights)
    if min_count < 1:
        raise ValueError("min_count must be >= 1")

    aligned = align_factor_matrices(specs, join=join)
    first = next(iter(aligned.values()))
    symbols = _symbols(first)
    joined = first.select("date")
    for spec in specs:
        values = aligned[spec.name]
        if spec.preprocessing is not None:
            values = prepare_factor(values, spec.preprocessing)
        if preprocessing is not None:
            values = prepare_factor(values, preprocessing)
        values = values.with_columns(
            pl.when(pl.col(symbols).is_finite()).then(pl.col(symbols)).otherwise(None)
        )
        renamed = values.rename({symbol: f"{spec.name}__{symbol}" for symbol in symbols})
        joined = joined.join(renamed, on="date", how="left")

    expressions = []
    for symbol in symbols:
        value_columns = [f"{spec.name}__{symbol}" for spec in specs]
        numerator_terms = [pl.col(column) * spec.weight for spec, column in zip(specs, value_columns, strict=True)]
        denominator_terms = [
            pl.when(pl.col(column).is_not_null()).then(abs(spec.weight)).otherwise(0.0)
            for spec, column in zip(specs, value_columns, strict=True)
        ]
        observation_count = pl.sum_horizontal(
            [pl.when(pl.col(column).is_not_null()).then(1).otherwise(0) for column in value_columns]
        )
        numerator = pl.sum_horizontal(numerator_terms)
        denominator = pl.sum_horizontal(denominator_terms)
        score = numerator / denominator if normalize else numerator
        expressions.append(
            pl.when(observation_count >= min_count)
            .then(pl.when((denominator == 0) & normalize).then(None).otherwise(score))
            .otherwise(None)
            .alias(symbol)
        )
    return joined.select([pl.col("date"), *expressions])


def factor_decay_linear(
    factors: pl.DataFrame,
    *,
    window: int,
    min_periods: int | None = None,
) -> pl.DataFrame:
    """Apply a trailing linear time decay to each symbol's factor series.

    Linear decay is a fixed filter -- the newest observation in the window gets
    weight ``window``, the oldest gets 1 -- so it is a weighted rolling sum
    rather than a per-window callback. Dividing by the weight mass actually
    present renormalizes around gaps, matching a plain weighted mean.
    """
    if window < 1:
        raise ValueError("window must be >= 1")
    if min_periods is None:
        min_periods = window
    if not 1 <= min_periods <= window:
        raise ValueError("min_periods must be in [1, window]")

    data = _factor_matrix_pl(factors)
    symbols = _symbols(data)
    if not symbols:
        return data
    weights = [float(index) for index in range(1, window + 1)]
    value = pl.col(symbols)
    present = value.is_not_null() & value.is_finite()
    weighted_sum = pl.when(present).then(value).otherwise(0.0).rolling_sum(
        window_size=window, weights=weights, min_samples=1)
    weight_mass = present.cast(pl.Float64).rolling_sum(window_size=window, weights=weights, min_samples=1)
    observed = present.cast(pl.Float64).rolling_sum(window_size=window, min_samples=1)
    return data.select("date", pl.when((observed < min_periods) | (weight_mass == 0))
                       .then(None).otherwise(weighted_sum / weight_mass).name.keep())


def factor_decay_exponential(
    factors: pl.DataFrame,
    *,
    span: float | None = None,
    halflife: float | None = None,
    min_periods: int = 1,
) -> pl.DataFrame:
    """Apply trailing exponential time decay to each symbol's factor series as Polars."""
    if span is None and halflife is None:
        raise ValueError("span or halflife is required")
    if span is not None and span <= 0:
        raise ValueError("span must be > 0")
    if halflife is not None and halflife <= 0:
        raise ValueError("halflife must be > 0")
    if min_periods < 1:
        raise ValueError("min_periods must be >= 1")
    data = _factor_matrix_pl(factors)
    return data.select(
        "date", pl.col(_symbols(data)).ewm_mean(
            span=span, half_life=halflife, min_samples=min_periods, adjust=False)
    )


def _normalize_factor_specs(
    factors: Mapping[str, object] | list[FactorSpec],
    *,
    weights: Mapping[str, float] | None = None,
) -> list[FactorSpec]:
    if isinstance(factors, Mapping):
        return [
            FactorSpec(name=name, values=values, weight=1.0 if weights is None else weights.get(name, 1.0))
            for name, values in factors.items()
        ]
    if weights is not None:
        return [replace(spec, weight=weights.get(spec.name, spec.weight)) for spec in factors]
    return list(factors)


def _factor_matrix_pl(frame) -> pl.DataFrame:
    data = _to_polars_matrix(frame).sort("date")
    symbols = sorted(_symbols(data))
    return data.select("date", pl.col(symbols).cast(pl.Float64, strict=False))


def _symbols(frame: pl.DataFrame) -> list[str]:
    return [column for column in frame.columns if column != "date"]
