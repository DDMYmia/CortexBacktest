from dataclasses import replace

import polars as pl
import pytest

from cortexbacktest.pipelines.factor_walk_forward import (
    FactorWalkForwardConfig, run_factor_walk_forward,
)
from cortexbacktest.pipelines.factor_workflow import run_factor_workflow
from test_factor_workflow import CONFIG, panel


SPLIT = FactorWalkForwardConfig(20, 10, 2, embargo_sessions=1)


def run(frame, split=SPLIT):
    return run_factor_walk_forward(frame, ["signal", "junk0"], split=split,
                                   label_definition="open[t+1] to open[t+2]",
                                   config=replace(CONFIG, resamples=50))


def test_purging_and_day_level_folds() -> None:
    result = run(panel(sessions=52, names=60, loading=0.02))
    assert len(result.folds) == 3
    first = result.folds[0]
    assert first["train_end"] == 17
    assert first["test_start"] == 21
    assert first["purged_sessions"] == 2
    assert first["train_end"] + SPLIT.label_horizon_sessions < first["test_start"] - 1
    assert result.oos.daily.select("date", "factor").is_duplicated().sum() == 0
    assert result.oos.daily["date"].unique().sort().to_list() == list(range(21, 51))
    assert result.manifest["unused_tail_sessions"] == 1


def test_later_data_cannot_change_earlier_fold_or_its_daily_costs() -> None:
    frame = panel(sessions=52, names=60, loading=0.02)
    original = run(frame)
    changed = run(frame.with_columns(
        pl.when(pl.col("date") >= 31).then(-pl.col("signal")).otherwise(pl.col("signal")).alias("signal")))
    assert original.folds[0] == changed.folds[0]
    assert original.oos.daily.filter(pl.col("date") < 31).equals(
        changed.oos.daily.filter(pl.col("date") < 31))


def test_combined_oos_recomputes_verdict_and_keeps_boundary_costs() -> None:
    frame = panel(sessions=52, names=60, loading=0.02)
    result = run(frame)
    expected = run_factor_workflow(frame.filter(pl.col("date").is_between(21, 50)),
                                   ["signal", "junk0"], config=replace(CONFIG, resamples=50))
    assert result.oos.evaluations == expected.evaluations
    assert result.oos.family == expected.family
    assert result.oos.daily.drop("fold").equals(expected.daily)
    assert result.report() == run(frame).report()


def test_missing_oos_session_cannot_be_silently_bridged() -> None:
    frame = panel(sessions=52, names=60).with_columns(
        pl.when(pl.col("date") != 45).then(pl.col("signal")).alias("signal"))
    with pytest.raises(ValueError, match="missing sessions"):
        run(frame)


def test_purge_cannot_empty_training_window() -> None:
    with pytest.raises(ValueError, match="purging"):
        run(panel(sessions=52, names=60), replace(SPLIT, label_horizon_sessions=20))


def test_actual_dates_remain_dates_in_daily_output():
    from datetime import date, timedelta

    frame = panel(sessions=52, names=60)
    frame = frame.with_columns(pl.col("date").replace_strict(
        {i: date(2020, 1, 1) + timedelta(days=i) for i in range(52)}))
    result = run(frame)
    assert result.oos.daily.schema["date"] == pl.Date


def test_demo_labels_do_not_bridge_missing_symbol_sessions():
    from examples.factor_workflow_demo import features

    frame = pl.DataFrame({"date": [0, 1, 2, 3, 0, 2, 3],
                          "symbol": ["A"] * 4 + ["B"] * 3,
                          "open": [10.0, 11.0, 12.0, 13.0, 10.0, 12.0, 13.0]})
    for name in ("close", "realized_variance", "dollar_volume", "close_volume_share",
                 "max_minute_move", "rv_per_minute", "vwap"):
        frame = frame.with_columns(pl.lit(1.0).alias(name))
    result = features(frame.lazy()).collect()
    assert result.filter((pl.col("symbol") == "B") & (pl.col("date") == 0))["forward_return"][0] is None
    assert result.filter((pl.col("symbol") == "A") & (pl.col("date") == 0))["forward_return"][0] is not None
