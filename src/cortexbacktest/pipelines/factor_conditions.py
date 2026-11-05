"""Availability-bound condition diagnostics on an existing OOS factor study."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import polars as pl

from .factor_walk_forward import FactorWalkForwardResult


@dataclass(frozen=True, slots=True)
class FactorConditionResult:
    thresholds: pl.DataFrame
    daily: pl.DataFrame
    summary: pl.DataFrame


def diagnose_factor_condition(
    study: FactorWalkForwardResult,
    states: pl.DataFrame,
    *,
    name: str,
    fixed_threshold: float | None = None,
    train_quantile: float = 0.5,
    min_train_observations: int = 20,
    min_state_runs: int = 5,
) -> FactorConditionResult:
    """Annotate existing returns without reselecting factors or resetting costs.

    States contain date, value, known_at and decision_at (aware datetimes).
    Callers supply the version that was available at each decision, and use
    trailing, not forward realised variance for tradable variance-premium states.
    High means strictly above the fixed or training-only quantile threshold.
    Missing, late and nonfinite states remain unknown. Summaries are descriptive;
    run counts are not a claim that regimes are independent observations.
    """
    required = {"date", "value", "known_at", "decision_at"}
    if not required <= set(states.columns):
        raise ValueError(f"missing state columns: {sorted(required - set(states.columns))}")
    if not name or not 0 <= train_quantile <= 1 or min_train_observations < 1 or min_state_runs < 2:
        raise ValueError("invalid condition name, quantile or evidence minimum")
    if fixed_threshold is not None and not isfinite(fixed_threshold):
        raise ValueError("fixed_threshold must be finite")
    for column in ("known_at", "decision_at"):
        dtype = states.schema[column]
        if not isinstance(dtype, pl.Datetime) or dtype.time_zone is None:
            raise ValueError("state and decision timestamps must be timezone-aware")
    if states["date"].null_count() or states["date"].is_duplicated().any():
        raise ValueError("state dates must be nonnull and unique")
    states = states.select(*sorted(required)).with_columns(
        (pl.col("value").is_finite() & (pl.col("known_at") <= pl.col("decision_at")))
        .fill_null(False).alias("available"))
    thresholds, annotated = [], []
    for fold in study.folds:
        training = states.filter(pl.col("date").is_between(pl.lit(fold["train_start"]), pl.lit(fold["train_end"]))
                                  & pl.col("available"))
        threshold = fixed_threshold
        if threshold is None and training.height >= min_train_observations:
            threshold = training["value"].quantile(train_quantile)
        thresholds.append({"fold": fold["fold"], "threshold": threshold,
                           "train_observations": training.height})
        rows = study.oos.daily.filter(pl.col("fold") == fold["fold"]).join(states, on="date", how="left")
        rows = rows.with_columns(
            pl.when(pl.col("available").fill_null(False) & pl.lit(threshold is not None))
            .then(pl.when(pl.col("value") > pl.lit(threshold, dtype=pl.Float64))
                  .then(pl.lit("high")).otherwise(pl.lit("low")))
            .otherwise(pl.lit("unknown")).alias("state"))
        annotated.append(rows)
    daily = pl.concat(annotated).sort("factor", "date")
    # Fold transitions do not automatically count as new independent regimes.
    daily = daily.with_columns(
        (pl.col("state") != pl.col("state").shift(1).over("factor"))
        .fill_null(True).alias("run_start"))
    summary = daily.group_by("factor", "state").agg(
        pl.len().alias("sessions"), pl.col("run_start").sum().alias("runs"),
        pl.col("spread").mean().alias("mean_spread"),
        pl.col("net_spread").mean().alias("mean_net_spread"),
        pl.col("cost").sum().alias("total_cost"),
    ).with_columns(
        pl.lit(name).alias("condition"),
        pl.when((pl.col("state") == "unknown") | (pl.col("runs") < min_state_runs))
        .then(pl.lit("insufficient_evidence")).otherwise(pl.lit("descriptive_only")).alias("status"),
    ).sort("factor", "state")
    return FactorConditionResult(pl.DataFrame(thresholds), daily, summary)
