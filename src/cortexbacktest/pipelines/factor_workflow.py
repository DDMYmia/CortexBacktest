"""One repeatable pass from a point-in-time panel to a verdict on a factor family.

The pipeline exists because of a specific failure, observed rather than imagined:
an information-coefficient table looks like a result. Run eight factors, sort by
IC, and the top row reads as a discovery. In the run this module was written
from, the top row by IC had a *negative* long-short Sharpe, the best long-short
Sharpe had a break-even cost of 10.6bp against a realistic 5-15bp, and the whole
family failed every multiple-testing correction but the most permissive one.

So the numbers that qualify a factor are not optional outputs here. There is no
way to obtain a `FactorEvaluation` without its turnover and break-even cost, and
no way to obtain the evaluations without the family-level verdict computed over
all of them together. A single factor evaluated alone is still a family of one
and is still priced as a search.

What this does NOT do: choose the factors, choose the universe, or decide that a
surviving factor is tradable. It reports; the judgement stays with the reader.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from hashlib import sha256
from math import isfinite, sqrt
from typing import Mapping, Sequence, cast

import numpy as np
import polars as pl

from ..data.backend import normalize_date_column
from ..event.manifest import engine_code_digest
from ..research import (
    benjamini_hochberg, deflated_sharpe_ratio, evaluate_factor, reality_check,
    sharpe_haircut, superior_predictive_ability, t_test_mean_zero,
)

TRADING_DAYS = 252


@dataclass(frozen=True, slots=True)
class FactorWorkflowConfig:
    """Everything the verdict depends on, in one place and in the manifest.

    `cost_bps` is a one-way cost applied to realised turnover. It is not a
    modelled spread or impact curve: it is the number the reader is asserting,
    and the report states it so a later reader can disagree with it explicitly.
    """

    quantiles: int = 5
    min_names: int = 200
    cost_bps: float = 10.0
    resamples: int = 2000
    block_size: float = 5.0
    seed: int = 7
    alpha: float = 0.05
    #: The probability a deflated Sharpe must clear to count as established.
    deflated_threshold: float = 0.95
    #: Total configurations inspected, including candidates omitted from this run.
    trials: int | None = None
    #: Observations per year; choose 12 for monthly or 52 for weekly panels.
    periods_per_year: float = TRADING_DAYS
    #: What to do with an eligible factor row whose forward return is missing.
    #: "refuse" (default) stops the run. "exclude" drops such rows and records
    #: their count per factor in the manifest; it is a stated choice, because a
    #: missing label is most often a name that stopped trading, and dropping it
    #: removes exactly the outcome a survivor-only sample would omit.
    unresolved_labels: str = "refuse"

    def __post_init__(self) -> None:
        if isinstance(self.periods_per_year, bool) or not isfinite(self.periods_per_year) or self.periods_per_year <= 0:
            raise ValueError("periods_per_year must be finite and positive")
        if self.quantiles < 2:
            raise ValueError("quantiles must be at least 2")
        if self.min_names < self.quantiles:
            raise ValueError("a cross-section thinner than one name per quantile is not one")
        for name in ("cost_bps", "block_size"):
            value = getattr(self, name)
            if not isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not 0 < self.alpha < 1 or not 0 < self.deflated_threshold < 1:
            raise ValueError("alpha and deflated_threshold are probabilities")
        if self.trials is not None and (type(self.trials) is not int or self.trials < 1):
            raise ValueError("trials must be a positive integer")
        if self.unresolved_labels not in ("refuse", "exclude"):
            raise ValueError("unresolved_labels must be 'refuse' or 'exclude'")


@dataclass(frozen=True, slots=True)
class FactorEvaluation:
    """One factor, with the numbers that decide whether it is worth anything.

    `sign_disagreement` is the cheap tell this module was built around: a rank IC
    and a long-short spread of opposite sign mean the relationship lives in the
    middle of the cross-section while the tails do something else. Momentum does
    this routinely, and ranking by IC alone buys the losing side of it.
    """

    name: str
    sessions: int
    rank_ic: float
    ic_ir: float
    ic_ar1: float
    spread_annual: float
    spread_sharpe: float
    turnover: float
    breakeven_cost_bps: float
    net_spread_annual: float
    p_value: float
    sign_disagreement: bool

    @property
    def survives_costs(self) -> bool:
        return self.net_spread_annual > 0


@dataclass(frozen=True, slots=True)
class FactorWorkflowResult:
    evaluations: tuple[FactorEvaluation, ...]
    family: Mapping[str, str | float | bool | None]
    established: tuple[str, ...]
    manifest: Mapping[str, object]
    daily: pl.DataFrame = field(default_factory=pl.DataFrame)
    corrections: Mapping[str, Mapping[str, float | bool]] = field(default_factory=dict)

    def report(self) -> dict[str, object]:
        return {
            "factors": [asdict(one) for one in self.evaluations],
            "family": dict(self.family),
            "established": list(self.established),
            "manifest": dict(self.manifest),
            "corrections": {name: dict(values) for name, values in self.corrections.items()},
        }

    def summary(self) -> str:
        lines = [f"{len(self.evaluations)} factors, "
                 f"{self.evaluations[0].sessions if self.evaluations else 0} aligned sessions"]
        for one in sorted(self.evaluations, key=lambda e: -e.spread_sharpe):
            flag = " (IC and spread disagree)" if one.sign_disagreement else ""
            lines.append(f"  {one.name:<22} sharpe {one.spread_sharpe:+6.3f}  "
                         f"p {one.p_value:.4f}  turnover {one.turnover:6.1%}  "
                         f"breakeven {one.breakeven_cost_bps:6.1f}bp{flag}")
        lines.append(f"  family: {dict(self.family)}")
        lines.append(f"  established: {list(self.established) or 'nothing'}")
        return "\n".join(lines)


def _daily_turnover(frame: pl.DataFrame, quantiles: int) -> pl.DataFrame:
    """Mean one-way turnover of both legs, including entry from cash.

    The ordinal buckets match evaluate_factor; callers sort by date/symbol so
    tied values have a deterministic assignment. Cost is four times turnover
    times one-way bps (buy/sell on each of the two unit-notional legs).
    """
    ranked = frame.with_columns(
        (((pl.col("factor").rank("ordinal").over("date") - 1)
          * quantiles / pl.len().over("date")).floor() + 1).alias("_bucket")
    )
    ranked = ranked.with_columns(
        (pl.col("_bucket") == quantiles).cast(pl.Float64).alias("_long"),
        (pl.col("_bucket") == 1).cast(pl.Float64).alias("_short"),
    )
    weights = (ranked.with_columns(
        (pl.col("_long") / pl.col("_long").sum().over("date")
         - pl.col("_short") / pl.col("_short").sum().over("date")).alias("_w"))
        .pivot(on="symbol", index="date", values="_w").sort("date")
        .with_columns(pl.exclude("date").fill_null(0.0)))
    matrix = weights.drop("date").to_numpy()
    traded = np.abs(np.diff(matrix, axis=0, prepend=np.zeros((1, matrix.shape[1])))).sum(axis=1)
    return weights.select("date").with_columns(pl.Series("turnover", traded / 4))


def evaluate_one(frame: pl.DataFrame, *, config: FactorWorkflowConfig) -> tuple[FactorEvaluation, pl.DataFrame]:
    """Evaluate a single factor. Returns the evaluation and its daily spread."""
    report = evaluate_factor(frame, quantiles=config.quantiles)
    daily = (report.spread_by_date.join(report.rank_ic_by_date, on="date")
             .filter(pl.col("spread").is_finite() & pl.col("rank_ic").is_finite())
             .sort("date"))
    original_dates = frame.select("date").unique()
    dates = normalize_date_column(original_dates).with_columns(
        original_dates["date"].alias("_original_date"))
    daily = daily.join(dates, on="date").drop("date").rename({"_original_date": "date"})
    if daily.height < 2:
        raise ValueError("a factor needs at least two sessions to be evaluated")
    daily = daily.join(_daily_turnover(frame.join(daily.select("date"), on="date").sort("date", "symbol"),
                                      config.quantiles), on="date").sort("date")
    daily = daily.with_columns((4 * pl.col("turnover") * config.cost_bps / 1e4).alias("cost"))
    daily = daily.with_columns((pl.col("spread") - pl.col("cost")).alias("net_spread"))
    ic, spread = daily["rank_ic"], daily["spread"]

    ic_values, spread_values = ic.to_numpy(), spread.to_numpy()
    ic_std = float(ic_values.std())
    spread_std = float(spread_values.std())
    # Daily ICs are not automatically independent: a factor built from a rolling
    # window is autocorrelated, and if that carries into the IC then sqrt(252) is
    # wrong. Measured rather than assumed -- in the run this was written from it
    # turned out to be near zero, which is a fact about that panel, not a law.
    ar1 = float(np.corrcoef(ic_values[:-1], ic_values[1:])[0, 1]) if len(ic_values) > 2 else 0.0

    turnover = float(daily["turnover"].to_numpy().mean())
    mean_spread = float(spread_values.mean())
    cost_per_session = float(daily["cost"].to_numpy().mean())
    breakeven = (mean_spread / (4 * turnover) * 1e4) if turnover else float("inf")

    rank_ic = float(ic_values.mean())
    return FactorEvaluation(
        name="factor",
        sessions=len(spread_values),
        rank_ic=rank_ic,
        ic_ir=(rank_ic / ic_std * sqrt(config.periods_per_year)) if ic_std else 0.0,
        ic_ar1=ar1 if isfinite(ar1) else 0.0,
        spread_annual=mean_spread * config.periods_per_year,
        spread_sharpe=(mean_spread / spread_std * sqrt(config.periods_per_year)) if spread_std else 0.0,
        turnover=turnover,
        breakeven_cost_bps=breakeven,
        net_spread_annual=(mean_spread - cost_per_session) * config.periods_per_year,
        p_value=float(t_test_mean_zero(spread_values).p_value),
        sign_disagreement=bool(rank_ic * mean_spread < 0),
    ), daily


def _family_statistics(performance, p_values, period_sharpes, settings):
    names, length = performance.columns, performance.height
    trials = settings.trials if settings.trials is not None else len(names)
    if trials < len(names):
        raise ValueError("trials cannot be smaller than the candidate family")
    white = reality_check(performance, resamples=settings.resamples,
                          block_size=settings.block_size, seed=settings.seed)
    spa = superior_predictive_ability(performance, resamples=settings.resamples,
                                      block_size=settings.block_size, seed=settings.seed)
    fdr = benjamini_hochberg(p_values, alpha=settings.alpha * len(names) / trials)
    best = max(names, key=period_sharpes.__getitem__)
    corrections = {}
    for name, rejected in zip(names, fdr.rejected, strict=True):
        sharpe = period_sharpes[name]
        corrections[name] = {
            "deflated_sharpe": float(deflated_sharpe_ratio(sharpe, observations=length, trials=trials)) if sharpe > 0 else 0.0,
            "haircut_adjusted_p": float(sharpe_haircut(sharpe, trials=trials, observations=length).adjusted_p_value) if sharpe > 0 else 1.0,
            "bh_rejected": bool(rejected),
        }
    family = {"best": best, "white_p": float(white.p_value), "spa_p": float(spa.p_value),
              "deflated_sharpe": corrections[best]["deflated_sharpe"],
              "haircut_adjusted_p": corrections[best]["haircut_adjusted_p"],
              "bh_rejected": int(np.asarray(fdr.rejected).sum()), "aligned_sessions": length,
              "trials": trials, "complete": trials == len(names)}
    agreed = (family["complete"] and period_sharpes[best] > 0
              and white.p_value < settings.alpha and spa.p_value < settings.alpha
              and family["bh_rejected"] > 0)
    supported = tuple(name for name in names if agreed and corrections[name]["bh_rejected"]
                      and corrections[name]["deflated_sharpe"] >= settings.deflated_threshold
                      and corrections[name]["haircut_adjusted_p"] < settings.alpha)
    return family, corrections, supported


def evaluate_return_family(returns: pl.DataFrame, *, config: FactorWorkflowConfig):
    """Apply the existing family statistics to caller-aligned period returns.

    Each non-date column is one tried candidate's net performance difference
    against the caller's benchmark (zero for absolute returns). No price,
    factor or execution is recalculated here. Missing trials cannot yield a
    supported family. Statistical support is not trading authorization.
    """
    if config.trials is None:
        raise ValueError("Return family requires the explicit full search trial count")
    if "date" not in returns.columns or returns.width < 2 or returns.height < 2:
        raise ValueError("Return family needs dates, candidates and at least two periods")
    if returns["date"].null_count() or returns["date"].n_unique() != returns.height:
        raise ValueError("Return family requires unique nonnull dates")
    performance = returns.sort("date").drop("date")
    if any(not column.is_finite().fill_null(False).all() for column in performance):
        raise ValueError("Return family requires finite complete observations")
    p_values, sharpes = [], {}
    for values in performance:
        p_values.append(t_test_mean_zero(values).p_value)
        deviation = cast(float | None, values.std(ddof=1))
        sharpes[values.name] = float(cast(float, values.mean()) / deviation) if deviation else 0.0
    if all(performance[name].std(ddof=1) == 0 for name in performance.columns):
        return {"family": {"status": "undefined_constant_returns", "complete": config.trials == performance.width,
                           "trials": config.trials, "aligned_sessions": performance.height},
                "corrections": {}, "statistically_supported": ()}
    family, corrections, supported = _family_statistics(performance, p_values, sharpes, config)
    return {"family": family, "corrections": corrections, "statistically_supported": supported}


def run_factor_workflow(
    panel: pl.DataFrame,
    factors: Sequence[str],
    *,
    config: FactorWorkflowConfig | None = None,
    date_col: str = "date",
    symbol_col: str = "symbol",
    forward_return_col: str = "forward_return",
    data_id: str = "unspecified",
) -> FactorWorkflowResult:
    """Evaluate a family of factors and price the search that produced it.

    `panel` is long-form: one row per (date, symbol), one column per factor, and
    a `forward_return` the caller has already aligned. This function does not
    build the forward return, because how it is built -- which price, which lag --
    is the single most consequential choice in the whole exercise and must not be
    a default buried here.
    """
    settings = config or FactorWorkflowConfig()
    missing = [name for name in (date_col, symbol_col, forward_return_col, *factors)
               if name not in panel.columns]
    if missing:
        raise ValueError(f"panel is missing columns: {missing}")
    if not factors:
        raise ValueError("a family needs at least one factor")
    if len(set(factors)) != len(factors):
        raise ValueError("factor names must be unique")
    trials = settings.trials if settings.trials is not None else len(factors)
    if trials < len(factors):
        raise ValueError("trials cannot be smaller than the candidate family")

    wide = panel.rename({date_col: "date", symbol_col: "symbol",
                         forward_return_col: "forward_return"}).sort("date", "symbol")
    if wide.select(pl.any_horizontal(pl.col("date", "symbol").is_null()).any()).item():
        raise ValueError("date and symbol must not be null")
    if wide.select(pl.struct("date", "symbol").is_duplicated().any()).item():
        raise ValueError("duplicate date/symbol identity")
    dense = wide.group_by("date").len().filter(pl.col("len") >= settings.min_names)
    wide = wide.join(dense.select("date"), on="date", how="inner")
    if wide.is_empty():
        raise ValueError(f"no session has {settings.min_names} names; the panel is too thin")

    frames, coverage = {}, {}
    common = wide.select("date").unique()
    raw_sessions = panel[date_col].n_unique()
    for name in factors:
        unresolved = wide.filter(pl.col(name).is_finite().fill_null(False)
                                 & ~pl.col("forward_return").is_finite().fill_null(False))
        if unresolved.height and settings.unresolved_labels == "refuse":
            raise ValueError(
                f"factor {name!r} has {unresolved.height} eligible rows with missing/nonfinite "
                "forward returns; resolve label coverage and terminal outcomes before evaluation"
            )
        one = wide.select("date", "symbol", pl.col(name).alias("factor"),
                          "forward_return").drop_nulls()
        if one.is_empty():
            raise ValueError(f"factor {name!r} is null on every row of the panel")
        one = one.filter(pl.col("factor").is_finite() & pl.col("forward_return").is_finite())
        valid = (one.group_by("date").agg(pl.len().alias("n"),
                 pl.col("factor").n_unique().alias("distinct"),
                 pl.col("forward_return").n_unique().alias("returns"))
                 .filter((pl.col("n") >= settings.min_names)
                         & (pl.col("distinct") >= settings.quantiles)
                         & (pl.col("returns") > 1)).select("date"))
        coverage[name] = {"raw_sessions": raw_sessions,
                          "valid_sessions": valid.height,
                          "excluded_unresolved_labels": unresolved.height}
        frames[name] = one
        common = common.join(valid, on="date")
    if common.height < 2:
        raise ValueError("fewer than two common valid sessions; factor panel is too thin")

    evaluations, daily_rows = [], []
    performance = common.sort("date")
    for name in factors:
        one = frames[name].join(common, on="date").sort("date", "symbol")
        evaluation, daily = evaluate_one(one, config=settings)
        evaluations.append(replace(evaluation, name=name))
        performance = performance.join(daily.select("date", pl.col("spread").alias(name)), on="date")
        daily_rows.append(daily.with_columns(pl.lit(name).alias("factor")))
        coverage[name]["common_sessions"] = common.height
    performance = performance.sort("date").drop("date")
    p_values = [one.p_value for one in evaluations]

    family, corrections, supported = _family_statistics(
        performance, p_values,
        {one.name: one.spread_sharpe / sqrt(settings.periods_per_year) for one in evaluations}, settings)
    established = tuple(one.name for one in evaluations if one.name in supported
                        and one.survives_costs and not one.sign_disagreement)

    digest = sha256(panel.select(date_col, symbol_col, *factors, forward_return_col)
                    .rename({date_col: "date", symbol_col: "symbol", forward_return_col: "forward_return"})
                    .sort("date", "symbol").hash_rows().to_numpy().tobytes()).hexdigest()
    manifest = {
        "workflow": "factor_family_v2",
        "polars_version": pl.__version__,
        "code_digest": engine_code_digest(),
        "data_digest": digest,
        "data_id": data_id,
        "factors": list(factors),
        "config": asdict(settings),
        "rows": panel.height,
        "sessions": int(wide["date"].n_unique()),
        "names": int(wide["symbol"].n_unique()),
        "coverage": coverage,
        "evaluation_scope": "common_valid_dates",
        "search_scope": "White/SPA cover supplied returns; BH counts omitted trials conservatively as p=1",
        "cost_model": "both_legs_target_weight_changes_including_initial_entry",
        "cost_limitations": "diagnostic target weights; no drift, borrow or terminal liquidation",
    }
    return FactorWorkflowResult(tuple(evaluations), family, established, manifest,
                                pl.concat(daily_rows).sort("date", "factor"), corrections)
