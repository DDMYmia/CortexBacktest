from datetime import datetime, timedelta, timezone

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from cortexbacktest.research import MiningFactorSpec, RevisionPanel, auto_mine_factors, mine_factors
from cortexbacktest.research.factor_expression import parse_expression
from cortexbacktest.research.preprocessing import rank_cross_section
from cortexbacktest.pipelines.mining_store import load_mining_result, save_mining_result
from cortexbacktest import load_backtest_dataset
from cortexbacktest.pipelines import run_mining_execution_pipeline


def panel():
    dates = [datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(days=i) for i in range(90)]
    return pl.DataFrame({"date": dates, **{s: [100 * (1 + r) ** i for i in range(90)]
                       for s, r in zip("ABCD", (.001, .002, .003, .004), strict=True)}})


@pytest.mark.parametrize("expression", ["__import__('os')", "momentum.shift(-1)", "momentum + sma_ratio",
    "rank(momentum, sma_ratio)", "rank(x=momentum)", "div(momentum,momentum)",
    "rank(" * 7 + "momentum" + ")" * 7, "rank(" + "x" * 600 + ")"])
def test_invalid_or_unbounded_expressions_fail_before_evaluation(expression):
    with pytest.raises(ValueError):
        MiningFactorSpec(expression, 5)


def test_canonical_identity_deduplicates_commutative_formulas():
    a = MiningFactorSpec("add(rank(momentum), rank(sma_ratio))", 5)
    b = MiningFactorSpec("add(rank(sma_ratio),rank(momentum))", 5)
    assert a == b and a.name == b.name
    assert parse_expression(a.indicator).leaves == {"momentum", "sma_ratio"}
    result = mine_factors(panel(), [a, b])
    assert result.trials[1, "status"] == "duplicate"


def test_ranking_masks_late_inputs_before_they_affect_other_assets():
    prices = panel()
    dates = prices["date"]
    availability = pl.DataFrame({"date": dates, **{s: dates for s in "ABCD"}}).with_columns(
        (pl.col("D") + timedelta(days=100)).alias("D"))
    spec = MiningFactorSpec("rank(momentum)", 5)
    actual = spec.build(prices, availability)
    expected = rank_cross_section(MiningFactorSpec("momentum", 5).build(prices, availability))
    assert_frame_equal(actual, expected)
    assert actual[10, "D"] is None
    assert actual[10, "C"] == 1.  # Not 3/4: D was excluded before ranking.
    revisions = prices.unpivot(index="date", variable_name="symbol", value_name="price").with_columns(
        pl.when(pl.col("symbol") == "D").then(pl.col("date") + timedelta(days=100))
        .otherwise(pl.col("date")).alias("recorded_at"), pl.col("date").alias("known_at"))
    received = RevisionPanel(revisions, prices.select("date", pl.col("date").alias("decision_time")))
    assert_frame_equal(received.factor(spec), expected)


def test_formula_arithmetic_is_causal_and_zero_division_stays_missing():
    prices = panel()
    spec = MiningFactorSpec("div(momentum,rolling_volatility)", 5)
    assert spec.history_points == 6
    assert_frame_equal(spec.build(prices.head(30)), spec.build(prices).head(30))
    flat = prices.with_columns(pl.lit(100.).alias("A"))
    assert spec.build(flat)["A"].null_count() == prices.height


def test_missing_interior_price_is_excluded_before_cross_sectional_ranking():
    prices = panel()
    missing = prices.with_columns(pl.when(pl.col("date") == prices[8, "date"])
                                 .then(None).otherwise(pl.col("D")).alias("D"))
    spec = MiningFactorSpec("rank(rolling_volatility)", 5)
    actual = spec.build(missing)
    native = MiningFactorSpec("rolling_volatility", 5).build(prices).slice(10, 1).with_columns(
        pl.lit(None, dtype=pl.Float64).alias("D"))
    assert_frame_equal(actual.slice(10, 1), rank_cross_section(native))


def test_automatic_formula_search_and_frozen_identity_use_existing_paths(tmp_path):
    prices = panel()
    result = auto_mine_factors(prices, research_direction="momentum", windows=(5,), budget=10)
    assert result.trials.height == 6
    assert sum(name.startswith("add(") for name in result.trials["candidate"]) == 3
    spec = MiningFactorSpec("add(rank(momentum),rank(sma_ratio))", 5)
    frozen = mine_factors(prices.head(60), [spec])
    assert frozen.selected == (spec,)
    receipt = save_mining_result(frozen, tmp_path / "selection")
    restored = load_mining_result(tmp_path / "selection", manifest_sha256=receipt)
    assert restored.selected == (spec,)
    assert_frame_equal(restored.selected[0].build(prices), spec.build(prices))
    raw = prices.unpivot(index="date", variable_name="symbol", value_name="close").with_columns(
        *(pl.col("close").alias(c) for c in ("open", "high", "low")), pl.lit(1_000_000.).alias("volume"))
    dataset = load_backtest_dataset(bars=raw, factors=raw.select("date", "symbol").with_columns(pl.lit(0.).alias("factor")))
    replay = run_mining_execution_pipeline(restored, dataset, execution_bars=raw)[spec.name]
    assert replay.event.fills
    assert min(fill.timestamp for fill in replay.event.fills) > frozen.inner_end
