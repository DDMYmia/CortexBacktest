"""Purged walk-forward selection, out-of-sample validation and cost sensitivity."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import json
import math
from pathlib import Path
from statistics import NormalDist
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import polars as pl

from ..contracts import Bar, FillPrice
from ..engine import BacktestEngine
from ..event.broker import BrokerSimulator
from ..event.feed import ColumnarBarFeed
from ..event.optimization import optimize_strategy
from ..research.validation import benjamini_hochberg, deflated_sharpe_ratio, purged_walk_forward_splits
from ..reporting.performance import returns_from_equity


@dataclass(frozen=True, slots=True)
class CostScenario:
    name: str
    commission_bps: float = 0.0
    slippage_bps: float = 0.0

    def __post_init__(self):
        if not self.name or any(not math.isfinite(v) or v < 0 for v in (self.commission_bps, self.slippage_bps)):
            raise ValueError("cost scenarios need a name and finite non-negative costs")


@dataclass(frozen=True, slots=True)
class RobustnessResult:
    selections: pl.DataFrame
    out_of_sample: pl.DataFrame
    cost_sensitivity: pl.DataFrame
    summary: dict[str, Any]

    def write_dir(self, path: str | Path) -> Path:
        directory = Path(path)
        directory.mkdir(parents=True, exist_ok=True)
        self.selections.write_parquet(directory / "selections.parquet")
        self.out_of_sample.write_parquet(directory / "out_of_sample.parquet")
        self.cost_sensitivity.write_parquet(directory / "cost_sensitivity.parquet")
        (directory / "summary.json").write_text(json.dumps(self.summary, indent=2, allow_nan=False) + "\n")
        return directory


def _hac_mean_test(values: np.ndarray, lag: int) -> float | None:
    """Two-sided asymptotic normal p-value using Bartlett/Newey-West variance."""
    n = len(values)
    if n < 3:
        return None
    centered = values - values.mean()
    variance = float(centered @ centered) / n
    for k in range(1, min(lag, n - 1) + 1):
        variance += 2 * (1 - k / (lag + 1)) * float(centered[k:] @ centered[:-k]) / n
    if variance <= 0:
        return None
    z = float(values.mean()) / math.sqrt(variance / n)
    return 2 * NormalDist().cdf(-abs(z))


def _neighborhood(rows, best, grid, metric):
    neighbors = []
    for row in rows:
        changed = [name for name in grid if row[name] != best[name]]
        if len(changed) == 1:
            name = changed[0]
            values = list(dict.fromkeys(grid[name]))
            if abs(values.index(row[name]) - values.index(best[name])) == 1:
                neighbors.append(float(row[metric]))
    return {"neighbor_count": len(neighbors),
            "neighbor_median": float(np.median(neighbors)) if neighbors else None,
            "neighbor_worst": min(neighbors) if neighbors else None}


def run_robustness_study(
    bars: Sequence[Bar] | ColumnarBarFeed,
    strategy_factory: Callable,
    parameter_grid: Mapping[str, Sequence[Any]],
    *,
    train_sessions: int,
    test_sessions: int,
    label_horizon: int = 1,
    embargo: int = 0,
    step_sessions: int | None = None,
    expanding: bool = True,
    costs: Sequence[CostScenario] = (CostScenario("baseline"), CostScenario("cost_5bps", 5, 0), CostScenario("cost_10bps", 5, 5)),
    initial_cash: float = 100_000,
    metric: str = "final_equity",
    periods_per_year: int = 252,
) -> RobustnessResult:
    """Choose parameters on training data only and evaluate fresh test engines.

    Splits operate on sessions, never individual symbol rows. Test windows must
    not overlap. Each fold starts with fresh cash and a cold strategy; no training
    positions or hidden strategy state leak into testing. Baseline cost is costs[0].
    Cost scenarios evaluate the same training-selected parameters; they do not
    reselect on test data. Neighborhood diagnostics also use training scores only.
    """
    if step_sessions is not None and step_sessions < test_sessions:
        raise ValueError("test windows must not overlap")
    if initial_cash <= 0 or not math.isfinite(initial_cash) or periods_per_year < 1:
        raise ValueError("initial_cash and periods_per_year must be positive")
    if not costs or len({cost.name for cost in costs}) != len(costs):
        raise ValueError("cost scenario names must be nonempty and unique")
    source = bars if isinstance(bars, ColumnarBarFeed) else ColumnarBarFeed.from_bars(bars)
    by_day: dict[date, list[Bar]] = {}
    for group in source:
        by_day.setdefault(group.timestamp.date(), []).extend(group.bars)
    days = sorted(by_day)
    splits = purged_walk_forward_splits(days, train_size=train_sessions, test_size=test_sessions,
                                       label_horizon=label_horizon, embargo=embargo,
                                       step_size=step_sessions, expanding=expanding)
    if not splits:
        raise ValueError("not enough sessions for a train/test split")
    selections, observations, cost_rows, fold_pvalues = [], [], [], []
    all_returns = []
    # These are dynamic strategy constructor keywords, not optimizer options.
    grid: dict[str, Any] = {key: list(values) for key, values in parameter_grid.items()}
    baseline = costs[0]
    def engine(cost):
        return BacktestEngine(broker=BrokerSimulator(initial_cash=initial_cash, fill_price=FillPrice.OPEN,
                              commission_bps=cost.commission_bps, slippage_bps=cost.slippage_bps),
                              trade_on_close=False, snapshot_frequency="session", result_retention="summary")
    for fold, (train, test) in enumerate(splits):
        if not len(train):
            raise ValueError("purging removed all training observations")
        train_feed = ColumnarBarFeed.from_bars((bar for index in train for bar in by_day[days[index]]), assume_sorted=True)
        test_feed = ColumnarBarFeed.from_bars((bar for index in test for bar in by_day[days[index]]), assume_sorted=True)
        search = optimize_strategy(train_feed, strategy_factory, maximize=metric,
                                   parallel_backend="serial", engine_factory=lambda: engine(baseline), **grid)
        best = search.best_row()
        if best is None:
            raise ValueError("parameter grid yielded no cases")
        selected = {name: best[name] for name in grid}
        selections.append({"fold": fold, "train_start": days[train[0]], "train_end": days[train[-1]],
                           "test_start": days[test[0]], "test_end": days[test[-1]],
                           "parameters_json": json.dumps(selected, sort_keys=True),
                           "train_score": best[metric], "trials": len(search.rows),
                           **_neighborhood(search.rows, best, grid, metric)})
        for scenario in costs:
            result = engine(scenario).run(test_feed, strategy_factory(**selected))
            curve = result.performance.equity_curve()
            equities = curve["equity"].to_numpy()
            previous = np.r_[initial_cash, equities[:-1]]
            if (previous <= 0).any():
                raise ValueError("non-positive equity prevents valid return statistics")
            returns = returns_from_equity(curve, initial_equity=initial_cash)["return"].to_numpy()
            cost_rows.append({"fold": fold, "scenario": scenario.name,
                              "commission_bps": scenario.commission_bps, "slippage_bps": scenario.slippage_bps,
                              "total_return": float(equities[-1] / initial_cash - 1),
                              "fills": result.fill_count, "commission": result.total_commission,
                              "slippage": result.total_slippage})
            if scenario is baseline:
                all_returns.extend(returns)
                fold_pvalues.append(_hac_mean_test(returns, max(label_horizon, 1)))
                for timestamp, value in zip(curve["timestamp"], returns, strict=True):
                    observations.append({"fold": fold, "timestamp": timestamp, "return": float(value)})
    returns = np.asarray(all_returns, dtype=float)
    deviation = float(returns.std(ddof=1)) if len(returns) > 1 else 0.0
    period_sharpe = float(returns.mean()) / deviation if deviation > 0 else None
    trials = sum(row["trials"] for row in selections)
    dsr = None
    if period_sharpe is not None and len(returns) >= 3:
        z = (returns - returns.mean()) / returns.std()
        dsr = deflated_sharpe_ratio(period_sharpe, observations=len(returns), trials=trials,
                                   skewness=float(np.mean(z ** 3)), kurtosis=float(np.mean(z ** 4)))
    qvalues = benjamini_hochberg([np.nan if p is None else p for p in fold_pvalues]).q_values
    for row, pvalue, qvalue in zip(selections, fold_pvalues, qvalues, strict=True):
        row.update(oos_hac_p_value=pvalue, oos_fold_q_value=float(qvalue) if np.isfinite(qvalue) else None)
    return RobustnessResult(pl.DataFrame(selections), pl.DataFrame(observations), pl.DataFrame(cost_rows), {
        "folds": len(splits), "test_observations": len(returns), "training_trials": trials,
        "baseline_cost": baseline.name, "mean_oos_return": float(returns.mean()),
        "annualized_oos_sharpe": None if period_sharpe is None else period_sharpe * math.sqrt(periods_per_year),
        "deflated_sharpe_probability": dsr,
        "positive_return_fraction": float(np.mean(returns > 0)),
        "assumptions": ["fresh cash and cold strategy per fold", "next-bar-open execution",
                        "no overlapping test windows", "HAC p-values use an asymptotic normal approximation",
                        "BH family comprises selected out-of-sample folds", "DSR uses all training trials as a conservative trial count",
                        "neighbor adjacency follows parameter grid order"],
    })
