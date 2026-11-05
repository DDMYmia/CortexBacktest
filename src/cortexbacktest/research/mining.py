"""Bounded, Polars-native factor discovery with separate selection and holdout APIs.

Paper-inspired evaluation baseline, not a reproduction of an LLM/RL miner.
Only causal native indicators are admitted; no generated Python is executed.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field, replace
from datetime import date, datetime
import hashlib
import math
from typing import cast
from itertools import combinations
from functools import lru_cache

import polars as pl

from ..search import SearchObservation, SurrogateOptimizer
from .factor_expression import BUILDERS, parse_expression
from .indicators import _native_expression, _price_matrix_pl
from .workflow import FactorResearchReport, run_factor_research, top_bottom_weights, _execution_factors
from .validation import WalkForwardSplit
from .revisions import RevisionPanel

@dataclass(frozen=True, slots=True)
class MiningFactorSpec:
    indicator: str
    window: int
    direction: int = 1

    def __post_init__(self):
        object.__setattr__(self, "indicator", parse_expression(self.indicator).name)
        if type(self.window) is not int or self.window < 2:
            raise ValueError("window must be an integer >= 2")
        if type(self.direction) is not int or self.direction not in (-1, 1):
            raise ValueError("direction must be -1 or 1")

    @property
    def name(self) -> str:
        return f"{self.indicator}:{self.window}:{self.direction}"

    @property
    def history_points(self) -> int:
        """Prices required by the builder, including the first return endpoint."""
        return self.window + bool(parse_expression(self.indicator).leaves &
                                  {"momentum", "rolling_volatility", "risk_adjusted_momentum"})

    def build(self, prices: pl.DataFrame, availability: pl.DataFrame | None = None) -> pl.DataFrame:
        if self.indicator not in BUILDERS:
            return self.compose(lambda component: component.build(prices, availability))
        ordered = _price_matrix_pl(prices)
        factor = ordered.select("date", self._expression(ordered.columns[1:]))
        availability = _availability(factor, availability)
        if availability is None:
            return factor
        decision = factor["date"].cast(pl.Datetime("us")).dt.epoch("us")
        masks = []
        for name in (name for name in factor.columns if name != "date"):
            known = availability[name].dt.epoch("us")
            if self.indicator == "momentum":
                eligible = (known <= decision) & (known.shift(self.window) <= decision)
            else:
                eligible = known.rolling_max(window_size=self.history_points, min_samples=self.history_points) <= decision
            masks.append(pl.when(eligible.fill_null(False)).then(factor[name]).otherwise(None).alias(name))
        return factor.with_columns(masks)

    def _expression(self, symbols: list[str]) -> pl.Expr:
        """One independently scoped window; no fitted state or earlier history."""
        column = pl.col(symbols)
        present = column.is_not_null() & column.is_finite()
        count = self.history_points
        eligible = (present & present.shift(self.window) if self.indicator == "momentum" else
                    present.cast(pl.UInt32).rolling_sum(window_size=count, min_samples=count) == count)
        value = _native_expression(self.indicator, column, self.window) * self.direction
        return pl.when(eligible.fill_null(False)).then(value).otherwise(None).name.keep()

    def compose(self, component: Callable[[MiningFactorSpec], pl.DataFrame]) -> pl.DataFrame:
        """Combine already causal leaves on their common decision grid."""
        return parse_expression(self.indicator).build(
            lambda name: component(replace(self, indicator=name, direction=1))
        ).with_columns(pl.exclude("date") * self.direction)


@dataclass(frozen=True, slots=True)
class MiningResult:
    selected: tuple[MiningFactorSpec, ...]
    trials: pl.DataFrame
    inner_sha256: str
    inner_end: date | datetime
    horizon: int
    cost_bps: float
    settings: dict = field(default_factory=dict)
    inner_availability_sha256: str | None = None
    inner_revision_sha256: str | None = None
    inner_execution_sha256: str | None = None


def _digest(frame: pl.DataFrame) -> str:
    return hashlib.sha256(frame.write_csv().encode()).hexdigest()


def _panel(prices: pl.DataFrame) -> pl.DataFrame:
    if "date" not in prices.columns or prices.width < 4 or prices.is_empty():
        raise ValueError("A dated panel with at least three assets is required")
    if prices.schema["date"] != pl.Date and not isinstance(prices.schema["date"], pl.Datetime):
        raise ValueError("date must be Date or Datetime")
    if prices["date"].null_count() or prices["date"].n_unique() != prices.height:
        raise ValueError("Dates must be present and unique")
    if prices.select(((pl.exclude("date") <= 0) | ~pl.exclude("date").is_finite()).any()).row(0).count(True):
        raise ValueError("Observed prices must be finite and positive; missing values may be null")
    return prices.sort("date")


def _availability(prices: pl.DataFrame, availability: pl.DataFrame | None) -> pl.DataFrame | None:
    if availability is None:
        return None
    if set(availability.columns) != set(prices.columns):
        raise ValueError("Availability must have the same dates and assets as prices")
    availability = availability.select(prices.columns).sort("date")
    if not availability["date"].equals(prices["date"]):
        raise ValueError("Availability must have the same dates and assets as prices")
    if any(not isinstance(availability.schema[name], pl.Datetime) for name in prices.columns if name != "date"):
        raise ValueError("Availability values must be Datetime timestamps")
    return availability


def _known_prices(prices: pl.DataFrame, availability: pl.DataFrame | None) -> pl.DataFrame:
    """Labels/valuation may use only versions received by this evaluation cutoff."""
    if availability is None:
        return prices
    cutoff = prices["date"].cast(pl.Datetime("us")).dt.epoch("us")[-1]
    return prices.with_columns([
        pl.when(availability[name].dt.epoch("us") <= cutoff).then(prices[name]).otherwise(None).alias(name)
        for name in prices.columns if name != "date"
    ])


def _dependence(left: pl.DataFrame, right: pl.DataFrame, minimum_dates: int) -> float:
    pairs = left.unpivot(index="date", variable_name="symbol", value_name="left").join(
        right.unpivot(index="date", variable_name="symbol", value_name="right"),
        on=["date", "symbol"],
    ).filter(pl.col("left").is_finite() & pl.col("right").is_finite()).sort("date", "symbol")
    daily = pairs.group_by("date", maintain_order=True).agg(
        pl.corr("left", "right", method="spearman").alias("rho"), pl.len().alias("n"),
    ).filter((pl.col("n") >= 3) & pl.col("rho").is_finite())
    # Too little overlap cannot establish independence.
    return abs(cast(float, daily["rho"].mean())) if daily.height >= minimum_dates else 1.0


def _bottom_quantile(mode: str) -> float | None:
    if mode not in ("long_short", "long_only"):
        raise ValueError("Unknown factor portfolio mode")
    return None if mode == "long_only" else 0.2


def mine_factors(
    inner_prices: pl.DataFrame | RevisionPanel, candidates: Iterable[MiningFactorSpec], *, horizon: int = 1,
    cost_bps: float = 10.0, max_factors: int = 5, min_rank_ic: float = 0.0,
    max_correlation: float = 0.9, min_coverage: float = 0.5, minimum_dates: int = 20,
    min_net_return: float = 0.0, max_turnover: float | None = None,
    on_evaluated: Callable[[MiningFactorSpec, float | None], None] | None = None,
    validation: WalkForwardSplit | None = None,
    availability: pl.DataFrame | None = None,
    execution: RevisionPanel | None = None,
    study_start: date | datetime | None = None,
    portfolio_mode: str = "long_short",
) -> MiningResult:
    """Select only on supplied inner data; log every proposal, including duplicates.

    Callers supply a point-in-time eligible universe. IC selection is exploratory,
    not a multiple-testing-adjusted claim of profitability or significance.
    """
    bottom_quantile = _bottom_quantile(portfolio_mode)
    revisions = inner_prices if isinstance(inner_prices, RevisionPanel) else None
    if revisions is not None and availability is not None:
        raise ValueError("RevisionPanel already owns availability clocks")
    _validate_execution(revisions, execution)
    if revisions is not None and execution is None:
        _require_close_decisions(revisions)
    prices = _panel(revisions.prices() if revisions is not None else cast(pl.DataFrame, inner_prices))
    availability = _availability(prices, availability)
    if not candidates or any(type(n) is not int or n < 1 for n in (horizon, max_factors, minimum_dates)):
        raise ValueError("Candidates and positive integer evaluation sizes are required")
    if not (math.isfinite(cost_bps) and cost_bps >= 0 and math.isfinite(min_rank_ic)
            and 0 < max_correlation <= 1 and 0 <= min_coverage <= 1
            and math.isfinite(min_net_return) and min_net_return >= 0
            and (max_turnover is None or math.isfinite(max_turnover) and max_turnover >= 0)):
        raise ValueError("Invalid mining thresholds")
    # Keep the complete history for causal feature construction and its receipt
    # fingerprint. Only the declared study period allocates train/test windows.
    offset = 0
    if study_start is not None:
        first = prices[0, "date"]
        if (type(study_start) is not type(first)
                or isinstance(study_start, datetime) and isinstance(first, datetime)
                and (study_start.utcoffset() is None) != (first.utcoffset() is None)):
            raise ValueError("Study start must use the panel's date type and timezone awareness")
        if not first <= study_start <= prices[-1, "date"]:
            raise ValueError("Study start must fall within the supplied inner history")
        offset = prices.filter(pl.col("date") < study_start).height
    # Native builders have no fitted parameters. Earlier prices only warm up
    # causal indicators; feedback uses disjoint inner test windows exclusively.
    windows = [(offset, prices.height - offset)]
    if validation is not None:
        windows = [(offset + int(test[0]), len(test)) for _, test in validation.split(prices.slice(offset))]
        if not windows or any(size <= horizon for _, size in windows):
            raise ValueError("Inner validation requires windows longer than the label horizon")
        if any(start < previous + size for (previous, size), (start, _) in zip(windows, windows[1:])):
            raise ValueError("Inner validation windows must not overlap")
    # Share each fold's cutoff snapshot across all candidate evaluations.
    observed_windows = {start: (
        (execution if execution is not None else revisions).prices(
            cutoff=revisions.decisions[start + size - 1, "decision_time"]).slice(start, size)
        if revisions is not None else
        _known_prices(prices.slice(start, size), availability.slice(start, size) if availability is not None else None)
    ) for start, size in windows}
    # Cache only within this study; another receipt cutoff must never reuse it.
    # The fixed bound limits retained panels for user-supplied candidate streams.
    @lru_cache(maxsize=32)
    def build_factor(spec: MiningFactorSpec) -> pl.DataFrame:
        if spec.indicator not in BUILDERS:
            return spec.compose(build_factor)
        return revisions.factor(spec) if revisions is not None else spec.build(prices, availability)

    execution_prices = execution.prices() if execution is not None else None
    trials, factors, seen = [], {}, set()
    for spec in candidates:
        row = {"candidate": spec.name, "rank_ic": None, "coverage": None,
               "net_return": None, "turnover": None,
               "max_correlation": None, "status": "duplicate", "folds": []}
        trials.append(row)
        if spec.name in seen:
            continue
        seen.add(spec.name)
        factor = build_factor(spec)
        if execution_prices is not None:
            _, factor = _execution_factors(execution_prices, factor, cast(RevisionPanel, revisions).decisions)
        folds = []
        for start, size in windows:
            # Compute forward labels from this slice: the final horizon rows
            # cannot borrow prices from the next window or the final holdout.
            observed = observed_windows[start]
            report = run_factor_research(observed, factor.slice(start, size),
                                         horizons=(horizon,), transaction_cost_bps=cost_bps, bottom_quantile=bottom_quantile)
            daily = report.horizon_results[horizon].rank_ic_by_date
            usable = daily.filter(pl.col("rank_ic").is_finite())
            folds.append({"start": observed[0, "date"], "end": observed[-1, "date"],
                          "observations": usable.height,
                          "rank_ic": cast(float, usable["rank_ic"].mean()) if usable.height else None,
                          "coverage": report.coverage_mean,
                          "final_equity": report.final_equity if math.isfinite(report.final_equity) else None,
                          "turnover": report.turnover_mean})
        count = sum(f["observations"] for f in folds)
        ic = sum(f["rank_ic"] * f["observations"] for f in folds if f["rank_ic"] is not None) / count if count else None
        coverage = sum(f["coverage"] * size for f, (_, size) in zip(folds, windows, strict=True)) / sum(size for _, size in windows)
        row.update(rank_ic=ic, coverage=coverage, status="insufficient_evidence", folds=folds)
        if all(f["observations"] >= minimum_dates and f["coverage"] >= min_coverage
               and f["final_equity"] is not None and math.isfinite(f["turnover"]) for f in folds):
            # Each disjoint fold starts in cash. This is a compounded validation
            # score, not a continuous strategy NAV across skipped train periods.
            net_return = math.prod(f["final_equity"] for f in folds) - 1
            turnover = sum(f["turnover"] * size for f, (_, size) in zip(folds, windows, strict=True)) / sum(size for _, size in windows)
            row.update(net_return=net_return if math.isfinite(net_return) else None, turnover=turnover)
            row["status"] = "weak" if ic is None or ic <= min_rank_ic else "eligible"
            if row["status"] == "eligible":
                if not math.isfinite(net_return) or any(f["final_equity"] <= 0 for f in folds) or net_return <= min_net_return:
                    row["status"] = "cost_rejected"
                elif max_turnover is not None and turnover > max_turnover:
                    row["status"] = "turnover_rejected"
            if row["status"] == "eligible":
                factors[spec.name] = (spec, pl.concat([factor.slice(start, size) for start, size in windows]))
        if on_evaluated is not None:
            on_evaluated(spec, ic if row["status"] in {"eligible", "weak"} else None)
    if not trials:
        raise ValueError("Candidates must not be empty")
    selected = []
    for row in sorted((r for r in trials if r["status"] == "eligible"),
                      key=lambda r: (-r["rank_ic"], r["candidate"])):
        spec, factor = factors[row["candidate"]]
        correlation = max((_dependence(factor, factors[s.name][1], minimum_dates) for s in selected), default=0.0)
        row["max_correlation"] = correlation
        row["status"] = "redundant" if correlation >= max_correlation else "capacity"
        if correlation < max_correlation and len(selected) < max_factors:
            selected.append(spec)
            row["status"] = "selected"
    return MiningResult(tuple(selected), pl.DataFrame(trials), _digest(prices),
                        cast(date | datetime, prices["date"][-1]), horizon, cost_bps,
                        {"horizon": horizon, "cost_bps": cost_bps, "max_factors": max_factors,
                         "min_rank_ic": min_rank_ic, "max_correlation": max_correlation,
                         "min_coverage": min_coverage, "minimum_dates": minimum_dates,
                         "min_net_return": min_net_return, "max_turnover": max_turnover,
                         "portfolio_mode": portfolio_mode,
                         "return_basis": "execution_prints" if execution is not None else "close_to_close",
                         **({"study_start": study_start.isoformat()} if study_start is not None else {}),
                         "validation": asdict(validation) if validation is not None else None},
                        _digest(availability) if availability is not None else None,
                        revisions.fingerprint() if revisions is not None else None,
                        execution.fingerprint() if execution is not None else None)


# Research mechanism and factor sign are distinct: direction constrains the
# hypothesis family; the sign below is fixed before observing any returns.
RESEARCH_DIRECTIONS = {
    "momentum": (("momentum", "sma_ratio", "risk_adjusted_momentum"), 1),
    "mean_reversion": (("momentum", "sma_ratio", "rolling_zscore"), -1),
    "low_volatility": (("rolling_volatility",), -1),
}


def auto_mine_factors(
    inner_prices: pl.DataFrame | RevisionPanel, *, research_direction: str,
    windows: tuple[int, ...] = (5, 10, 20, 40, 60), budget: int = 12,
    seed: int = 7, horizon: int = 1, cost_bps: float = 10.0,
    min_net_return: float = 0.0, max_turnover: float | None = None,
    validation: WalkForwardSplit | None = None,
    availability: pl.DataFrame | None = None,
    execution: RevisionPanel | None = None,
    study_start: date | datetime | None = None,
    portfolio_mode: str = "long_short",
) -> MiningResult:
    """Automatically propose, evaluate, and refine within a selected mechanism.

    Reuses the existing surrogate optimizer. Only inner scores enter feedback;
    proposals failing evidence, net-return or turnover gates receive no IC credit.
    candidate identity and evaluation are shared with manual mining. Stops at
    the trial budget or exhaustion. This does not call an LLM or execute code.
    """
    if research_direction not in RESEARCH_DIRECTIONS:
        raise ValueError(f"Choose a supported research direction: {tuple(RESEARCH_DIRECTIONS)}")
    if type(budget) is not int or not 1 <= budget <= 1000 or type(seed) is not int:
        raise ValueError("budget must be 1..1000 and seed must be an integer")
    if not windows or len(set(windows)) != len(windows):
        raise ValueError("Distinct nonempty windows are required")
    indicators, sign = RESEARCH_DIRECTIONS[research_direction]
    indicators = (*indicators, *(parse_expression(f"add(rank({left}),rank({right}))").name
                                for left, right in combinations(indicators, 2)))
    for window in windows:
        MiningFactorSpec(indicators[0], window, sign)
    grid = {"indicator": indicators, "window": windows}
    observations: list[SearchObservation] = []
    optimizer = SurrogateOptimizer(warmup=3)

    def proposals():
        for _ in range(budget):
            params = optimizer.suggest(grid, observations, random_state=seed)
            if params is None:
                break
            yield MiningFactorSpec(params["indicator"], params["window"], sign)

    def observe(spec: MiningFactorSpec, score: float | None):
        observations.append(SearchObservation(
            {"indicator": spec.indicator, "window": spec.window},
            score if score is not None else -1.0,
        ))

    result = mine_factors(inner_prices, proposals(), horizon=horizon,
                         cost_bps=cost_bps, min_net_return=min_net_return, max_turnover=max_turnover,
                         on_evaluated=observe, validation=validation, availability=availability, execution=execution,
                         study_start=study_start, portfolio_mode=portfolio_mode)
    return replace(result, settings={**result.settings, "research_direction": research_direction,
                                    "windows": list(windows), "budget": budget, "seed": seed,
                                    "optimizer": dict(optimizer.configuration)})


def _frozen_inputs(result: MiningResult, prices: pl.DataFrame | RevisionPanel,
                   availability: pl.DataFrame | None, inner_execution: RevisionPanel | None):
    revisions = prices if isinstance(prices, RevisionPanel) else None
    if revisions is not None and availability is not None:
        raise ValueError("RevisionPanel already owns availability clocks")
    inner_revisions = revisions.through(result.inner_end) if revisions is not None else None
    if (inner_execution.fingerprint() if inner_execution is not None else None) != result.inner_execution_sha256:
        raise ValueError("Inner execution evidence changed since selection")
    if (inner_revisions.fingerprint() if inner_revisions is not None else None) != result.inner_revision_sha256:
        raise ValueError("Inner revision evidence or decision clocks changed since selection")
    prices = _panel(revisions.prices() if revisions is not None else cast(pl.DataFrame, prices))
    availability = _availability(prices, availability)
    inner_availability = availability.filter(pl.col("date") <= result.inner_end) if availability is not None else None
    digest = _digest(inner_availability) if inner_availability is not None else None
    if digest != result.inner_availability_sha256:
        raise ValueError("Inner availability changed since selection")
    inner = inner_revisions.prices() if inner_revisions is not None else prices.filter(pl.col("date") <= result.inner_end)
    if _digest(inner) != result.inner_sha256:
        raise ValueError("Inner data changed since selection")
    return revisions, prices, availability, inner_revisions


def frozen_factor_values(result: MiningResult, prices: pl.DataFrame | RevisionPanel, *,
                         availability: pl.DataFrame | None = None,
                         inner_execution: RevisionPanel | None = None,
                         factor: str | None = None, last_only: bool = False) -> dict[str, pl.DataFrame]:
    """Frozen decision features, retaining nulls before ranking turns them into cash."""
    revisions, values, availability, _ = _frozen_inputs(result, prices, availability, inner_execution)
    factors = {spec.name: (revisions.factor(spec, last_only=last_only) if revisions is not None else spec.build(values, availability))
               .filter(pl.col("date") > result.inner_end) for spec in result.selected
               if factor is None or spec.name == factor}
    if revisions is not None:
        clock = dict(zip(revisions.decisions["date"], revisions.decisions["decision_time"], strict=True))
        factors = {name: factor.with_columns(pl.col("date").replace_strict(clock)) for name, factor in factors.items()}
    return factors


def frozen_factor_targets(result: MiningResult, prices: pl.DataFrame | RevisionPanel, *,
                          availability: pl.DataFrame | None = None,
                          inner_execution: RevisionPanel | None = None) -> dict[str, pl.DataFrame]:
    """Compute selected factors' decision-time targets without future return labels.

    Restore result from its verified checkpoint first. The training execution
    panel, when bound by the study, covers only the frozen training decisions;
    no future opening print is required. Returned dates are actual decision times
    for revision inputs (otherwise the supplied price-grid dates),
    never authorization to fill at that same close. Ranking and long/short
    weights use the portfolio policy frozen during selection (top 20%, optional bottom 20%).
    """
    factors = frozen_factor_values(result, prices, availability=availability, inner_execution=inner_execution)
    return {name: top_bottom_weights(factor, bottom_quantile=_bottom_quantile(result.settings.get("portfolio_mode", "long_short")))
            for name, factor in factors.items()}


def evaluate_holdout(result: MiningResult, prices: pl.DataFrame | RevisionPanel, *, availability: pl.DataFrame | None = None,
                     execution: RevisionPanel | None = None) -> dict[str, FactorResearchReport]:
    """Evaluate frozen candidates on later dates, retaining causal warm-up history.

    No selection or parameter update occurs here. Persist and freeze the result
    before calling; the API cannot prevent a researcher from rerunning a study.
    """
    revisions = prices if isinstance(prices, RevisionPanel) else None
    if revisions is not None and availability is not None:
        raise ValueError("RevisionPanel already owns availability clocks")
    _validate_execution(revisions, execution)
    if revisions is not None and execution is None:
        _require_close_decisions(revisions)
    inner_execution = execution.head(revisions.through(result.inner_end).decisions.height) if execution is not None and revisions is not None else None
    revisions, prices, availability, inner_revisions = _frozen_inputs(result, prices, availability, inner_execution)
    if execution is not None:
        assert revisions is not None and inner_revisions is not None
        # Training may finish after the following session has already opened.
        # That print cannot be an out-of-sample trade after freezing selection.
        cutoff = inner_revisions.decisions[-1, "decision_time"]
        outer = execution.prices().filter(pl.col("date") > cutoff)
        if outer.height <= result.horizon:
            raise ValueError("Insufficient later holdout data")
        return {spec.name: run_factor_research(outer, revisions.factor(spec),
                    decision_times=revisions.decisions, horizons=(result.horizon,), transaction_cost_bps=result.cost_bps,
                    bottom_quantile=_bottom_quantile(result.settings.get("portfolio_mode", "long_short")))
                for spec in result.selected}
    outer = prices.filter(pl.col("date") > result.inner_end)
    if outer.height <= result.horizon:
        raise ValueError("Insufficient later holdout data")
    return {spec.name: run_factor_research(
        _known_prices(outer, availability.filter(pl.col("date") > result.inner_end) if availability is not None else None),
        (revisions.factor(spec) if revisions is not None else spec.build(prices, availability)).filter(pl.col("date") > result.inner_end),
        horizons=(result.horizon,), transaction_cost_bps=result.cost_bps,
        bottom_quantile=_bottom_quantile(result.settings.get("portfolio_mode", "long_short")),
    ) for spec in result.selected}


def _require_close_decisions(panel: RevisionPanel) -> None:
    if panel.decisions.filter(pl.col("date") != pl.col("decision_time")).height:
        raise ValueError("Mining requires close-time decisions; delayed decisions need execution-aligned labels")


def _validate_execution(features: RevisionPanel | None, execution: RevisionPanel | None) -> None:
    if execution is None:
        return
    if features is None or not features.decisions["decision_time"].equals(execution.decisions["decision_time"]):
        raise ValueError("Execution and feature panels require the same per-session evaluation cutoffs")
    if (execution.decisions["date"] >= features.decisions["date"]).any():
        raise ValueError("Each execution opening must precede its corresponding feature bar close")
    if set(features.revisions["symbol"]) != set(execution.revisions["symbol"]):
        raise ValueError("Execution and feature panels require the same assets")
