from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
import json

import numpy as np
import polars as pl
import pytest

from cortexbacktest.research import HoldingConfig, PersistentWeightRecipe, apply_holding_recipe

T = datetime(2025, 1, 1, tzinfo=timezone.utc)


def ts(i):
    return T + timedelta(days=i)


def config(**kw):
    return HoldingConfig(**({"enter_threshold": 0.5, "exit_threshold": -0.5} | kw))


def test_persistence_hysteresis_exit_confirmation_and_exact_boundaries():
    recipe = PersistentWeightRecipe(["A"], config=config(enter_periods=2, exit_periods=2))
    scores = [0.5, 0.49, 0.5, 0.5, 0.0, -0.5, -0.49, -0.5, -0.5]
    expected = [0, 0, 0, 1, 1, 1, 1, 1, 0]
    assert [
        recipe.update(ts(i), {"A": s}, {"A": 1.0})["A"] for i, s in enumerate(scores)
    ] == expected


def test_minimum_active_duration_and_cooldown_excludes_exact_number_of_decisions():
    recipe = PersistentWeightRecipe(["A"], config=config(min_active_periods=4, cooldown_periods=2))
    scores = [1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0]
    assert [recipe.update(ts(i), {"A": s}, {"A": 1.0})["A"] for i, s in enumerate(scores)] == [
        1,
        1,
        1,
        0,
        0,
        0,
        1,
    ]


def test_null_breaks_entry_streak_and_immediately_clears_active_cohorts_despite_min_hold():
    recipe = PersistentWeightRecipe(
        ["A"],
        config=config(
            enter_periods=2, min_active_periods=20, cooldown_periods=1, holding_periods=3
        ),
    )
    expected = [0, 0, 0, 1 / 3, 0, 0, 0, 1 / 3]
    scores = [1.0, None, 1.0, 1.0, None, 1.0, 1.0, 1.0]
    actual = [recipe.update(ts(i), {"A": s}, {"A": 1.0})["A"] for i, s in enumerate(scores)]
    assert actual == pytest.approx(expected)
    assert len(recipe.to_document()["cohorts"]) == 1  # no resurrection of pre-null cohort


def test_overlapping_baskets_have_fixed_budget_unused_cash_and_expiry():
    recipe = PersistentWeightRecipe(["B", "A"], config=config(holding_periods=3))
    targets = [{"A": 1.0, "B": 0.0}, *[{"A": 0.0, "B": 1.0}] * 3]
    actual = [
        recipe.update(ts(i), {"A": 1.0, "B": 1.0}, target) for i, target in enumerate(targets)
    ]
    assert actual[0] == pytest.approx({"A": 1 / 3, "B": 0})
    assert actual[1] == pytest.approx({"A": 1 / 3, "B": 1 / 3})
    assert actual[2] == pytest.approx({"A": 1 / 3, "B": 2 / 3})
    assert actual[3] == pytest.approx({"A": 0, "B": 1})
    assert all(sum(abs(v) for v in row.values()) <= 1 + 1e-14 for row in actual)


def test_exit_prunes_all_old_baskets_and_reentry_uses_only_new_budget():
    recipe = PersistentWeightRecipe(["A"], config=config(holding_periods=10, cooldown_periods=1))
    scores = [1.0, 1.0, -1.0, 1.0, 1.0]
    assert [
        recipe.update(ts(i), {"A": s}, {"A": 1.0})["A"] for i, s in enumerate(scores)
    ] == pytest.approx([0.1, 0.2, 0, 0, 0.1])
    assert recipe.to_document()["cohorts"][0]["formed_at"] == 4


def test_nondivisible_formation_and_holding_periods_do_not_overspend():
    recipe = PersistentWeightRecipe(["A"], config=config(holding_periods=3, formation_interval=2))
    assert [recipe.update(ts(i), {"A": 1.0}, {"A": 1.0})["A"] for i in range(7)] == [
        0.5,
        0.5,
        1.0,
        0.5,
        1.0,
        0.5,
        1.0,
    ]


def test_signed_cohorts_net_before_execution_and_never_increase_gross_budget():
    recipe = PersistentWeightRecipe(["A"], config=config(holding_periods=2, max_gross=0.8))
    assert recipe.update(ts(0), {"A": 1.0}, {"A": 0.8}) == {"A": 0.4}
    assert recipe.update(ts(1), {"A": 1.0}, {"A": -0.8}) == {"A": 0.0}
    assert recipe.update(ts(2), {"A": 1.0}, {"A": -0.8}) == {"A": -0.8}


def matrices(n=15):
    dates = [ts(i) for i in range(n)]
    targets = pl.DataFrame({"date": dates, "A": [0.6] * n, "B": [0.4] * n})
    scores = pl.DataFrame(
        {
            "date": dates,
            "A": [1.0, 1.0, 0.0, -1.0, None, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0, None, 1.0][
                :n
            ],
            "B": [1.0] * n,
        }
    )
    return targets, scores


def test_every_restart_boundary_matches_uninterrupted_state_and_weights():
    targets, scores = matrices()
    cfg = config(
        enter_periods=2, exit_periods=2, cooldown_periods=2, holding_periods=5, formation_interval=2
    )
    expected, checkpoint = apply_holding_recipe(targets, scores, config=cfg)
    for cut in range(1, targets.height):
        first, saved = apply_holding_recipe(targets.head(cut), scores.head(cut), config=cfg)
        # Simulate JSON persistence, not an in-memory object reuse.
        second, restored = apply_holding_recipe(
            targets.slice(cut),
            scores.slice(cut),
            config=cfg,
            checkpoint=json.loads(json.dumps(saved, allow_nan=False)),
        )
        assert pl.concat([first, second]).equals(expected)
        assert restored == checkpoint
    initial = PersistentWeightRecipe(["A", "B"], config=cfg).to_document()
    actual, state = apply_holding_recipe(
        targets.reverse(), scores.reverse(), config=cfg, checkpoint=initial
    )
    assert actual.equals(expected) and state == checkpoint


@pytest.mark.parametrize(
    "kind",
    [
        "nan",
        "infinity",
        "bool",
        "missing_symbol",
        "extra_symbol",
        "budget",
        "naive",
        "duplicate",
        "backward",
    ],
)
def test_invalid_event_is_rejected_before_mutating_state(kind):
    recipe = PersistentWeightRecipe(["A"], config=config())
    recipe.update(ts(0), {"A": 1.0}, {"A": 1.0})
    before = recipe.to_document()
    stamp = ts(1)
    scores = {"A": 1.0}
    targets = {"A": 1.0}
    if kind == "nan":
        scores = {"A": float("nan")}
    elif kind == "infinity":
        targets = {"A": float("inf")}
    elif kind == "bool":
        scores = {"A": True}
    elif kind == "missing_symbol":
        scores = {}
    elif kind == "extra_symbol":
        targets = {"A": 1.0, "B": 0.0}
    elif kind == "budget":
        targets = {"A": 1.1}
    elif kind == "naive":
        stamp = stamp.replace(tzinfo=None)
    elif kind == "duplicate":
        stamp = ts(0)
    elif kind == "backward":
        stamp = ts(-1)
    with pytest.raises(ValueError):
        recipe.update(stamp, scores, targets)
    assert recipe.to_document() == before


@pytest.mark.parametrize(
    "kind",
    [
        "version",
        "config",
        "universe",
        "index",
        "active_type",
        "counter",
        "age",
        "inactive_cohort",
        "expired",
        "budget",
        "timestamp",
        "extra",
    ],
)
def test_invalid_checkpoint_rejected(kind):
    cfg = config(holding_periods=3)
    recipe = PersistentWeightRecipe(["A"], config=cfg)
    recipe.update(ts(0), {"A": 1.0}, {"A": 1.0})
    doc = recipe.to_document()
    if kind == "version":
        doc["version"] = True
    elif kind == "config":
        doc["config"]["holding_periods"] = 4
    elif kind == "universe":
        doc["symbols"] = ["B"]
    elif kind == "index":
        doc["index"] = True
    elif kind == "active_type":
        doc["signals"]["A"]["active"] = 1
    elif kind == "counter":
        doc["signals"]["A"]["enter_count"] = 1
    elif kind == "age":
        doc["signals"]["A"]["active_age"] = 2
    elif kind == "inactive_cohort":
        doc["signals"]["A"] = {
            "active": False,
            "enter_count": 0,
            "exit_count": 0,
            "active_age": 0,
            "cooldown_left": 0,
        }
    elif kind == "expired":
        doc["index"] = 4
    elif kind == "budget":
        doc["cohorts"][0]["weights"]["A"] = 1.0
    elif kind == "timestamp":
        doc["last_timestamp"] = "2025-01-01T00:00:00"
    elif kind == "extra":
        doc["unknown"] = 1
    with pytest.raises(ValueError):
        PersistentWeightRecipe.from_document(doc, symbols=["A"], config=cfg)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"enter_periods": True},
        {"exit_periods": 0},
        {"min_active_periods": 0},
        {"cooldown_periods": -1},
        {"holding_periods": 0},
        {"formation_interval": 0},
        {"max_gross": float("nan")},
        {"max_gross": 0},
        {"enter_threshold": -0.5},
        {"exit_threshold": True},
    ],
)
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        config(**kwargs)


@pytest.mark.parametrize(
    "kind", ["null_target", "nan_score", "missing_date", "duplicate", "naive", "text", "nano"]
)
def test_batch_requires_explicit_aligned_clocks_and_values(kind):
    targets, scores = matrices(3)
    if kind == "null_target":
        targets = targets.with_columns(pl.lit(None, dtype=pl.Float64).alias("A"))
    elif kind == "nan_score":
        scores = scores.with_columns(pl.lit(float("nan")).alias("A"))
    elif kind == "missing_date":
        scores = scores.slice(1)
    elif kind == "duplicate":
        scores = pl.concat([scores, scores.head(1)])
    elif kind == "naive":
        scores = scores.with_columns(pl.col("date").dt.replace_time_zone(None))
    elif kind == "text":
        scores = scores.with_columns(pl.lit("1").alias("A"))
    elif kind == "nano":
        scores = scores.with_columns(
            pl.col("date").dt.cast_time_unit("ns") + pl.duration(nanoseconds=1)
        )
    with pytest.raises(ValueError):
        apply_holding_recipe(targets, scores, config=config())


def test_independent_bounded_cohort_oracle_for_random_signed_targets():
    rng = np.random.default_rng(104)
    cfg = config(holding_periods=7, formation_interval=3, max_gross=1.2)
    recipe = PersistentWeightRecipe(["A", "B"], config=cfg)
    history = []
    for i in range(50):
        values = rng.uniform(-1, 1, 2)
        values = values / sum(abs(values)) * 1.2
        if i % 3 == 0:
            history.append((i, values.copy() / 3))
        expected = sum((v for at, v in history if i - at < 7), start=np.zeros(2))
        actual = recipe.update(
            ts(i), {"A": 1.0, "B": 1.0}, dict(zip(["A", "B"], map(float, values), strict=True))
        )
        np.testing.assert_allclose([actual["A"], actual["B"]], expected, atol=1e-15)
        assert sum(abs(x) for x in actual.values()) <= 1.2 + 1e-14


def test_checkpoint_document_does_not_alias_live_cohort_or_signal_state():
    recipe = PersistentWeightRecipe(["A"], config=config(holding_periods=3))
    recipe.update(ts(0), {"A": 1.0}, {"A": 1.0})
    doc = recipe.to_document()
    expected = deepcopy(doc)
    doc["cohorts"][0]["weights"]["A"] = 0.0
    doc["signals"]["A"]["active"] = False
    assert recipe.to_document() == expected
    assert asdict(recipe.config) == expected["config"]


def test_native_bridge_trades_only_net_targets_and_charges_no_idle_sleeve_costs():
    from cortexbacktest import BacktestEngine, Bar, BrokerSimulator, ColumnarBarFeed
    from cortexbacktest.contracts import FillPrice
    from cortexbacktest.event.rebalance import RebalanceConfig, TargetWeightStrategy

    target = pl.DataFrame({"date": [ts(i) for i in range(4)], "A": [1.0, -1.0, 1.0, -1.0]})
    scores = target.with_columns(pl.lit(1.0).alias("A"))
    net, _ = apply_holding_recipe(target, scores, config=config(holding_periods=2))
    assert net["A"].to_list() == [0.5, 0.0, 0.0, 0.0]
    bars = [
        Bar("A", ts(i) + timedelta(hours=12), 100.0, 101.0, 99.0, 100.0, volume=10000.0)
        for i in range(4)
    ]
    openings = {bar.timestamp: bar.timestamp - timedelta(hours=11) for bar in bars}
    broker = BrokerSimulator(
        initial_cash=1000.0, fill_price=FillPrice.OPEN, allow_short=True, commission_bps=10
    )
    strategy = TargetWeightStrategy(
        net,
        broker=broker,
        config=RebalanceConfig(allow_fractional=True, cash_buffer=0.01),
        trade_at_open=True,
        decision_times=True,
    )
    result = BacktestEngine(broker=broker, trade_on_close=False, opening_times=openings).run(
        ColumnarBarFeed.from_bars(bars), strategy
    )
    fills = result.records.fills()
    assert fills.height == 2  # enter .5 net capital, exit; opposing later sleeves do not trade.
    assert fills["quantity"].to_list() == pytest.approx([4.95, 4.95])
    assert result.total_commission == pytest.approx(0.99)
    assert result.summary()["final_equity"] == pytest.approx(999.01)


def test_absent_decisions_are_not_invented_as_calendar_days():
    recipe = PersistentWeightRecipe(["A"], config=config(enter_periods=2))
    assert recipe.update(ts(0), {"A": 1.0}, {"A": 1.0}) == {"A": 0.0}
    assert recipe.update(ts(10), {"A": 1.0}, {"A": 1.0}) == {"A": 1.0}
    assert recipe.to_document()["index"] == 1


def test_recipe_universe_and_configuration_are_read_only():
    recipe = PersistentWeightRecipe(["A"], config=config())
    with pytest.raises(AttributeError):
        recipe.config = config(holding_periods=2)
    with pytest.raises(AttributeError):
        recipe.symbols = ("B",)
