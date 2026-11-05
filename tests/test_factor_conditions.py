from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from cortexbacktest.pipelines.factor_conditions import diagnose_factor_condition
from test_factor_walk_forward import run
from test_factor_workflow import panel


def states():
    decisions = [datetime(2020, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(52)]
    return pl.DataFrame({"date": range(52), "value": [float(i) for i in range(52)],
                         "known_at": decisions, "decision_at": decisions})


def test_threshold_uses_training_only_and_does_not_change_existing_returns():
    study = run(panel(sessions=52, names=60, loading=0.02))
    original = diagnose_factor_condition(study, states(), name="variance", min_train_observations=10)
    changed = diagnose_factor_condition(study, states().with_columns(
        pl.when(pl.col("date") >= 21).then(pl.col("value") * 1000)
        .otherwise(pl.col("value")).alias("value")), name="variance", min_train_observations=10)
    assert original.thresholds["threshold"][0] == changed.thresholds["threshold"][0]
    assert original.daily.sort("date", "factor").select(study.oos.daily.columns).equals(study.oos.daily)
    assert set(original.summary["status"]) == {"insufficient_evidence"}


def test_late_and_missing_states_stay_unknown():
    study = run(panel(sessions=52, names=60))
    data = states().filter(pl.col("date") != 23).with_columns(
        pl.when(pl.col("date") == 22).then(pl.col("known_at") + pl.duration(days=1))
        .otherwise(pl.col("known_at")).alias("known_at"))
    result = diagnose_factor_condition(study, data, name="inversion", fixed_threshold=0.0)
    assert set(result.daily.filter(pl.col("date").is_in([22, 23]))["state"]) == {"unknown"}


def test_naive_timestamps_and_duplicate_states_are_rejected():
    study = run(panel(sessions=52, names=60))
    with pytest.raises(ValueError, match="timezone-aware"):
        diagnose_factor_condition(study, states().with_columns(pl.col("known_at").dt.replace_time_zone(None)),
                                  name="bad")
    with pytest.raises(ValueError, match="unique"):
        diagnose_factor_condition(study, pl.concat([states(), states().head(1)]), name="bad")
