from datetime import datetime, timedelta, timezone
import json

import numpy as np
import polars as pl
import pytest

from cortexbacktest.adapters.qlib_predictions import _holding_targets
from cortexbacktest.research.holding import HoldingConfig, apply_holding_recipe

OPTIONS = dict(
    enter_periods=2,
    exit_periods=2,
    min_active_periods=3,
    cooldown_periods=2,
    holding_periods=3,
    formation_interval=1,
)


def targets():
    first = [1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    return pl.DataFrame(
        {
            "date": [
                datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(days=i) for i in range(10)
            ],
            "A": first,
            "B": [1 - x for x in first],
        }
    )


def test_membership_confirmation_cooldown_and_fixed_slot_cohorts_match_manual_path():
    out, checkpoint = _holding_targets(targets(), OPTIONS)
    expected = [
        [0.0, 0.0],
        [1 / 3, 0.0],
        [2 / 3, 0.0],
        [2 / 3, 0.0],
        [0.0, 1 / 3],
        [0.0, 1 / 3],
        [0.0, 0.0],
        [0.0, 0.0],
        [1 / 3, 0.0],
        [2 / 3, 0.0],
    ]
    np.testing.assert_allclose(out.select("A", "B"), expected)
    assert checkpoint["index"] == 9 and len(checkpoint["cohorts"]) == 2
    assert checkpoint["config"]["max_gross"] == 1.0


def test_json_checkpoint_resume_matches_uninterrupted_targets_and_state():
    data = targets()
    full, final = _holding_targets(data, OPTIONS)
    left, checkpoint = _holding_targets(data.head(6), OPTIONS)
    config = HoldingConfig(enter_threshold=1.0, exit_threshold=0.0, **OPTIONS)
    rest = data.slice(6)
    scores = rest.select(
        "date", (pl.col("A") > 0).cast(pl.Float64), (pl.col("B") > 0).cast(pl.Float64)
    )
    right, restored = apply_holding_recipe(
        rest, scores, config=config, checkpoint=json.loads(json.dumps(checkpoint))
    )
    assert pl.concat([left, right]).equals(full) and restored == final
    with pytest.raises(ValueError, match="strictly increasing"):
        apply_holding_recipe(data.head(1), data.head(1), config=config, checkpoint=checkpoint)
    changed = {**checkpoint, "config": {**checkpoint["config"], "cooldown_periods": 0}}
    with pytest.raises(ValueError, match="pinned configuration"):
        apply_holding_recipe(rest, scores, config=config, checkpoint=changed)


def test_future_membership_does_not_change_prefix_and_disabled_equivalent_default_is_identity():
    data = targets()
    before, _ = _holding_targets(data, OPTIONS)
    after, _ = _holding_targets(
        data.with_columns(
            pl.when(pl.int_range(pl.len()) >= 6).then(0.0).otherwise(pl.col("A")).alias("A")
        ),
        OPTIONS,
    )
    assert before.head(6).equals(after.head(6))
    default = dict(
        enter_periods=1,
        exit_periods=1,
        min_active_periods=1,
        cooldown_periods=0,
        holding_periods=1,
        formation_interval=1,
    )
    identity, _ = _holding_targets(data, default)
    assert identity.equals(data)


@pytest.mark.parametrize(
    "patch",
    [{"enter_periods": True}, {"holding_periods": 253}, {"formation_interval": 0}, {"extra": 1}],
)
def test_unbounded_or_ambiguous_counters_reject(patch):
    with pytest.raises(ValueError, match="bounded"):
        _holding_targets(targets(), {**OPTIONS, **patch})
