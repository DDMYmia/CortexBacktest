"""Parameter sweeps and stability reports for factor research."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Mapping, cast

import polars as pl

from ..data import Universe
from .preprocessing import FactorPreprocessConfig
from .workflow import FactorResearchReport, run_factor_research
from .validation import benjamini_hochberg, deflated_sharpe_ratio, t_test_mean_zero

if TYPE_CHECKING:
    from .indicator_registry import IndicatorRegistry


FactorFactory = Callable[[pl.DataFrame], pl.DataFrame]
ResearchReport = FactorResearchReport



def best_case_in(table: pl.DataFrame, metric: str) -> str | None:
    """The case with the highest finite `metric`; None when no case has one."""
    if table.is_empty():
        return None
    if metric not in table.columns:
        raise ValueError(f"Unknown metric: {metric}")
    scores = table.select(["case", metric]).filter(pl.col(metric).is_finite())
    if scores.is_empty():
        return None
    return str(scores.sort(metric, descending=True).item(0, "case"))


@dataclass(frozen=True, slots=True)
class FactorSweepCase:
    """One factor research configuration to evaluate."""

    name: str
    factor: pl.DataFrame | FactorFactory
    params: Mapping[str, object] = field(default_factory=dict)
    preprocessing: FactorPreprocessConfig | None = None
    top_quantile: float = 0.2
    bottom_quantile: float | None = 0.2
    transaction_cost_bps: float = 0.0
    indicator_metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class FactorSweepResult:
    """Collection of factor research reports from a parameter sweep."""

    reports: dict[str, ResearchReport]
    cases: dict[str, FactorSweepCase]

    def summary(self) -> pl.DataFrame:
        rows = []
        for name, report in self.reports.items():
            case = self.cases[name]
            summary = report.summary()
            for row in summary.iter_rows(named=True):
                horizon = row.get("horizon")
                rows.append(
                    {
                        "case": name,
                        "indicator_unit": case.indicator_metadata.get("output_unit", "unspecified"),
                        "indicator_warmup_bars": case.indicator_metadata.get("warmup_bars"),
                        "indicator_metadata": json.dumps(dict(case.indicator_metadata), sort_keys=True),
                        "horizon": horizon,
                        "mean_ic": row.get("mean_ic"),
                        "mean_rank_ic": row.get("mean_rank_ic"),
                        "mean_spread": row.get("mean_spread"),
                        "mean_coverage": float(report.coverage_mean),
                        "mean_turnover": float(report.turnover_mean),
                        "final_equity": float(report.final_equity),
                        **{f"param_{key}": value for key, value in case.params.items()},
                    }
                )
        if not rows:
            return pl.DataFrame(
                {
                    "case": [],
                    "horizon": [],
                    "mean_ic": [],
                    "mean_rank_ic": [],
                    "mean_spread": [],
                    "mean_coverage": [],
                    "mean_turnover": [],
                    "final_equity": [],
                }
            )
        return pl.DataFrame(rows).sort(["case", "horizon"])

    def stability(self) -> pl.DataFrame:
        """Summarize robustness across horizons for each case."""
        rows = []
        for name, report in self.reports.items():
            summary = report.summary()
            rank_ic = summary["mean_rank_ic"] if "mean_rank_ic" in summary.columns else pl.Series("mean_rank_ic", [], dtype=pl.Float64)
            spread = summary["mean_spread"] if "mean_spread" in summary.columns else pl.Series("mean_spread", [], dtype=pl.Float64)
            rows.append(
                {
                    "case": name,
                    "horizon_count": summary.height,
                    "avg_rank_ic": _series_mean(rank_ic),
                    "min_rank_ic": _series_min(rank_ic),
                    "positive_rank_ic_ratio": _positive_ratio(rank_ic),
                    "avg_spread": _series_mean(spread),
                    "min_spread": _series_min(spread),
                    "positive_spread_ratio": _positive_ratio(spread),
                    "mean_coverage": float(report.coverage_mean),
                    "mean_turnover": float(report.turnover_mean),
                    "final_equity": float(report.final_equity),
                }
            )
        if not rows:
            return pl.DataFrame(
                {
                    "case": [],
                    "horizon_count": [],
                    "avg_rank_ic": [],
                    "min_rank_ic": [],
                    "positive_rank_ic_ratio": [],
                    "avg_spread": [],
                    "min_spread": [],
                    "positive_spread_ratio": [],
                    "mean_coverage": [],
                    "mean_turnover": [],
                    "final_equity": [],
                }
            )
        return pl.DataFrame(rows).sort(
            ["positive_rank_ic_ratio", "avg_rank_ic", "final_equity"],
            descending=[True, True, True],
        )

    def best_case(self, *, metric: str = "avg_rank_ic") -> str | None:
        return best_case_in(self.stability(), metric)

    def statistical_summary_pl(self, *, periods_per_year: int = 252) -> pl.DataFrame:
        """Return sweep-level Sharpe diagnostics with a multiple-testing penalty.

        The number of cases is used as the effective trial count, so callers
        cannot accidentally present a selected sweep winner as a one-off test.
        Use ``WalkForwardSplit`` / ``purged_walk_forward_splits`` to construct
        out-of-sample inputs before calling a sweep.
        """
        if periods_per_year <= 0:
            raise ValueError("periods_per_year must be positive")
        rows = []
        for name, report in self.reports.items():
            returns = _equity_returns(report)
            observations = len(returns)
            deviation = cast(float | None, returns.std(ddof=1)) if observations > 1 else None
            period_sharpe = float(cast(float, returns.mean()) / deviation) if deviation else 0.0
            rows.append(
                {
                    "case": name,
                    "observations": observations,
                    "sharpe": period_sharpe * periods_per_year**0.5,
                    "deflated_sharpe_ratio": deflated_sharpe_ratio(period_sharpe, observations=observations, trials=len(self.reports)) if observations > 1 else 0.0,
                }
            )
        return pl.DataFrame(rows).sort("deflated_sharpe_ratio", descending=True) if rows else pl.DataFrame()

    def fdr_summary_pl(self, *, alpha: float = 0.05, metric: str = "rank_ic") -> pl.DataFrame:
        """Per-case IC significance with a false-discovery-rate control.

        Each case/horizon is tested for a non-zero mean IC, then Benjamini-
        Hochberg adjusts across the whole sweep: reading a sweep's best raw
        p-value as if it were a single test overstates significance. ``q_value``
        is the FDR-adjusted p-value and ``significant`` marks q <= ``alpha``.

        This complements ``statistical_summary_pl``, which deflates Sharpe for
        the same selection problem. Feed the sweep out-of-sample inputs
        (``WalkForwardSplit`` / ``purged_walk_forward_splits``) for the test to
        mean anything.
        """
        if metric not in {"ic", "rank_ic"}:
            raise ValueError("metric must be 'ic' or 'rank_ic'")
        rows = []
        for name, report in self.reports.items():
            for horizon, result in report.horizon_results.items():
                test = t_test_mean_zero(_metric_by_date(result, metric))
                rows.append(
                    {
                        "case": name,
                        "horizon": horizon,
                        f"mean_{metric}": test.mean,
                        "observations": test.observations,
                        "t_stat": test.t_stat,
                        "p_value": test.p_value,
                    }
                )
        if not rows:
            return pl.DataFrame(
                schema={
                    "case": pl.String,
                    "horizon": pl.Int64,
                    f"mean_{metric}": pl.Float64,
                    "observations": pl.Int64,
                    "t_stat": pl.Float64,
                    "p_value": pl.Float64,
                    "q_value": pl.Float64,
                    "significant": pl.Boolean,
                }
            )
        frame = pl.DataFrame(rows)
        adjusted = benjamini_hochberg(frame["p_value"].to_numpy(), alpha=alpha)
        return frame.with_columns(
            pl.Series("q_value", adjusted.q_values),
            pl.Series("significant", adjusted.rejected),
        ).sort(["q_value", "case", "horizon"], nulls_last=True)

def _metric_by_date(result, metric: str):
    """IC series for one horizon, from either the pandas or Polars report."""
    values = getattr(result, f"{metric}_by_date")
    if isinstance(values, pl.DataFrame):
        if values.is_empty():
            return []
        column = metric if metric in values.columns else values.columns[-1]
        return values[column].to_numpy()
    return values.to_numpy()


def run_factor_sweep(
    prices,
    cases: list[FactorSweepCase],
    *,
    universe: Universe | pl.DataFrame | None = None,
    horizons: tuple[int, ...] = (1, 5, 20),
    quantiles: int = 5,
) -> FactorSweepResult:
    """Run factor sweep configurations through the Polars research workflow."""
    if not cases:
        raise ValueError("cases must not be empty")

    reports = {}
    case_map = {}
    for case in cases:
        if case.name in case_map:
            raise ValueError(f"Duplicate factor sweep case: {case.name}")
        factor = case.factor(prices) if callable(case.factor) else case.factor
        reports[case.name] = run_factor_research(
            prices,
            factor,
            universe=universe,
            preprocessing=case.preprocessing,
            horizons=horizons,
            quantiles=quantiles,
            top_quantile=case.top_quantile,
            bottom_quantile=case.bottom_quantile,
            transaction_cost_bps=case.transaction_cost_bps,
        )
        case_map[case.name] = case
    return FactorSweepResult(reports=reports, cases=case_map)


def factor_stability_report(report: ResearchReport) -> pl.DataFrame:
    """Create a single-factor stability summary across horizons."""
    summary = report.summary()
    rank_ic = summary["mean_rank_ic"] if "mean_rank_ic" in summary.columns else pl.Series("mean_rank_ic", [], dtype=pl.Float64)
    spread = summary["mean_spread"] if "mean_spread" in summary.columns else pl.Series("mean_spread", [], dtype=pl.Float64)
    return pl.DataFrame(
        [{
            "horizon_count": summary.height,
            "avg_rank_ic": _series_mean(rank_ic),
            "min_rank_ic": _series_min(rank_ic),
            "positive_rank_ic_ratio": _positive_ratio(rank_ic),
            "avg_spread": _series_mean(spread),
            "min_spread": _series_min(spread),
            "positive_spread_ratio": _positive_ratio(spread),
            "mean_coverage": float(report.coverage_mean),
            "mean_turnover": float(report.turnover_mean),
            "final_equity": float(report.final_equity),
        }]
    )


def _equity_returns(report: ResearchReport) -> pl.Series:
    """Period returns from a research equity curve."""
    frame = report.equity_curve_frame()
    if frame.is_empty() or "equity" not in frame.columns:
        return pl.Series("return", [], dtype=pl.Float64)
    return frame.select(pl.col("equity").pct_change().alias("return")).drop_nulls()["return"]


def _series_mean(values) -> float:
    clean = _clean_series(values)
    if len(clean) == 0:
        return float("nan")
    return float(clean.mean())


def _series_min(values) -> float:
    clean = _clean_series(values)
    if len(clean) == 0:
        return float("nan")
    return float(clean.min())


def _positive_ratio(values) -> float:
    clean = _clean_series(values)
    if len(clean) == 0:
        return float("nan")
    return float((clean > 0).mean())


def _clean_series(values):
    if isinstance(values, pl.Series):
        return values.filter(values.is_finite())
    return values.dropna()


def build_indicator_sweep_cases(
    indicator_name: str,
    param_grid: Mapping[str, list[object]],
    *,
    base_name: str | None = None,
    preprocessing: FactorPreprocessConfig | None = None,
    registry: "IndicatorRegistry | None" = None,
) -> list[FactorSweepCase]:
    """Generate FactorSweepCases from an indicator name and parameter grid.

    Allows automated parameter optimization sweeps across indicator parameters.
    """
    from itertools import product
    from .indicator_registry import INDICATORS

    reg = registry or INDICATORS
    prefix = base_name or indicator_name
    keys = list(param_grid.keys())
    value_lists = [param_grid[k] for k in keys]

    cases: list[FactorSweepCase] = []
    for combo in product(*value_lists):
        param_dict: dict[str, Any] = dict(zip(keys, combo))
        param_slug = "_".join(f"{k}{v}" for k, v in param_dict.items())
        case_name = f"{prefix}_{param_slug}" if param_slug else prefix

        cases.append(
            FactorSweepCase(
                name=case_name,
                factor=lambda prices, p=param_dict: reg.run(indicator_name, prices, **p),
                params=param_dict,
                indicator_metadata=reg.describe(indicator_name, **param_dict),
                preprocessing=preprocessing,
            )
        )
    return cases
