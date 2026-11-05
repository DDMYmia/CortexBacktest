"""Purged session splits for a frozen, precomputed family of factor candidates.

No fitted transforms or candidate selection occur here. A caller must construct
features causally; splitting cannot repair features fitted on the full sample.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from hashlib import sha256
from typing import Sequence

import polars as pl

from ..research.validation import purged_walk_forward_splits
from .factor_workflow import FactorWorkflowConfig, FactorWorkflowResult, run_factor_workflow


@dataclass(frozen=True, slots=True)
class FactorWalkForwardConfig:
    train_sessions: int
    test_sessions: int
    label_horizon_sessions: int
    embargo_sessions: int = 0
    expanding: bool = True

    def __post_init__(self) -> None:
        if self.train_sessions < 2 or self.test_sessions < 2:
            raise ValueError("train/test windows need at least two sessions")
        if self.label_horizon_sessions < 1 or self.embargo_sessions < 0:
            raise ValueError("label horizon must be positive and embargo nonnegative")


@dataclass(frozen=True, slots=True)
class FactorWalkForwardResult:
    folds: tuple[dict[str, object], ...]
    oos: FactorWorkflowResult
    manifest: dict[str, object]

    def report(self) -> dict[str, object]:
        return {"folds": list(self.folds), "oos": self.oos.report(),
                "daily_oos": self.oos.daily.to_dicts(), "manifest": self.manifest}

    def summary(self) -> str:
        return f"{len(self.folds)} retrospective OOS folds\n{self.oos.summary()}"


def run_factor_walk_forward(
    panel: pl.DataFrame,
    factors: Sequence[str],
    *,
    split: FactorWalkForwardConfig,
    label_definition: str,
    config: FactorWorkflowConfig | None = None,
    data_id: str = "unspecified",
) -> FactorWalkForwardResult:
    """Evaluate every frozen candidate on contiguous, nonoverlapping test folds.

    ``label_horizon_sessions`` includes execution lag through the label's final
    observation. Dates must cover the trading sessions, including rows with
    missing features. Irregular per-symbol label endpoints require a different
    splitter and are not supported by this fixed-horizon interface.

    The combined evaluation runs once on all OOS dates, so costs cross fold
    boundaries without resetting to cash. Missing common OOS dates fail closed:
    joining across an unobserved session would invent a continuous portfolio.
    """
    if not label_definition.strip():
        raise ValueError("label_definition must describe the price, lag and horizon")
    days = panel.select("date").unique().sort("date")["date"]
    splits = purged_walk_forward_splits(
        days, train_size=split.train_sessions, test_size=split.test_sessions,
        label_horizon=split.label_horizon_sessions, embargo=split.embargo_sessions,
        expanding=split.expanding,
    )
    if not splits:
        raise ValueError("not enough sessions for a train/test split")
    settings = config or FactorWorkflowConfig()
    folds, tests = [], []
    for number, (train, test) in enumerate(splits):
        if len(train) < 2:
            raise ValueError("purging leaves fewer than two training sessions")
        train_days, test_days = days.gather(train), days.gather(test)
        training = run_factor_workflow(
            panel.filter(pl.col("date").is_in(train_days.implode())), factors,
            config=settings, data_id=data_id,
        )
        tests.append(pl.DataFrame({"date": test_days, "fold": number}))
        folds.append({
            "fold": number, "train_start": train_days[0], "train_end": train_days[-1],
            "test_start": test_days[0], "test_end": test_days[-1],
            "train_sessions": len(train), "test_sessions": len(test),
            "purged_sessions": (int(test[0]) - split.embargo_sessions
                                if split.expanding else split.train_sessions) - len(train),
            "train": training.report(),
        })
    test_dates = pl.concat(tests)
    testing = panel.join(test_dates.select("date"), on="date")
    oos = run_factor_workflow(testing, factors, config=settings, data_id=data_id)
    if oos.daily["date"].n_unique() != test_dates.height:
        raise ValueError("OOS common coverage has missing sessions; continuous costs are undefined")
    daily = oos.daily.join(test_dates, on="date").sort("date", "factor")
    oos = replace(oos, daily=daily)
    for fold in folds:
        rows = daily.filter(pl.col("fold") == fold["fold"])
        fold["test"] = rows.group_by("factor").agg(
            pl.len().alias("sessions"), pl.col("rank_ic").mean(),
            pl.col("spread").mean().alias("mean_spread"),
            pl.col("turnover").mean(), pl.col("cost").sum().alias("total_cost"),
            pl.col("net_spread").mean().alias("mean_net_spread"),
        ).sort("factor").to_dicts()
    manifest = {
        "workflow": "factor_walk_forward_v1", "split": asdict(split),
        "label_definition": label_definition, "factors": list(factors),
        "config": asdict(settings), "data_id": data_id,
        "oos_data_digest": oos.manifest["data_digest"],
        "input_data_digest": sha256(panel.select("date", "symbol", *factors, "forward_return")
                                     .sort("date", "symbol").hash_rows().to_numpy().tobytes()).hexdigest(),
        "code_digest": oos.manifest["code_digest"],
        "selection": "none; all candidates and directions frozen before testing",
        "evidence": "retrospective_oos; not an untouched holdout or PIT certification",
        "cost_model": oos.manifest["cost_model"],
        # Under `unresolved_labels="exclude"` this is where the dropped rows are
        # counted, per factor; under "refuse" every count is zero.
        "oos_coverage": oos.manifest["coverage"],
        "unused_tail_sessions": len(days) - int(splits[-1][1][-1]) - 1,
    }
    return FactorWalkForwardResult(tuple(folds), oos, manifest)
