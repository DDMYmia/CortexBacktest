from datetime import datetime, timedelta, timezone
from math import sin
from statistics import stdev

import polars as pl
import pytest

from cortexbacktest import load_backtest_dataset
from cortexbacktest.pipelines import run_mining_execution_pipeline
from cortexbacktest.pipelines.mining_store import load_mining_result, save_mining_result
from cortexbacktest.research import MiningFactorSpec, RevisionPanel, auto_mine_factors, mine_factors


def prices(size=120):
    data = {"date": [datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(days=i) for i in range(size)]}
    for index, symbol in enumerate("ABCD", 1):
        values = [100.]
        for day in range(1, size):
            values.append(values[-1] * (1 + .002 * index + .001 * sin(day)))
        data[symbol] = values
    return pl.DataFrame(data)


def test_formula_matches_independent_arithmetic_and_does_not_read_future():
    panel = prices(12)
    spec = MiningFactorSpec("risk_adjusted_momentum", 5)
    result = spec.build(panel)
    for name in "ABCD":
        values = panel[name].to_list()
        expected = (values[5] / values[0] - 1) / stdev(values[i] / values[i - 1] - 1 for i in range(1, 6))
        assert result[5, name] == pytest.approx(expected)
    assert spec.build(panel.head(7)).equals(result.head(7))
    assert spec.history_points == 6
    assert spec.build(panel.with_columns(pl.lit(100.).alias("A")))["A"].null_count() == panel.height


def test_middle_window_invalidation_and_late_receipt_mask_composed_signal():
    panel = prices(12)
    dates = panel["date"].to_list()
    spec = MiningFactorSpec("risk_adjusted_momentum", 5)
    missing = panel.with_columns(pl.when(pl.col("date") == dates[2]).then(None).otherwise(pl.col("A")).alias("A"))
    assert spec.build(missing)[5, "A"] is None
    availability = pl.DataFrame({"date": dates, **{s: dates for s in "ABCD"}}).with_columns(
        pl.when(pl.col("date") == dates[2]).then(pl.lit(dates[6])).otherwise(pl.col("A")).alias("A"))
    assert spec.build(panel, availability)[5, "A"] is None
    assert spec.build(panel, availability)[6, "A"] == spec.build(panel)[6, "A"]
    revisions = panel.unpivot(index="date", variable_name="symbol", value_name="price").with_columns(
        pl.col("date").alias("recorded_at"), pl.col("date").alias("known_at"),
        pl.lit(None, dtype=pl.String).alias("invalid_reason"))
    invalid = revisions.filter((pl.col("date") == dates[2]) & (pl.col("symbol") == "A")).with_columns(
        pl.lit(dates[4]).alias("recorded_at"), pl.lit(None, dtype=pl.Float64).alias("price"),
        pl.lit("unverified_price_basis").alias("invalid_reason"))
    causal = RevisionPanel(pl.concat([revisions, invalid]), panel.select("date", pl.col("date").alias("decision_time")))
    assert causal.factor(spec)[5, "A"] is None


def test_composed_candidate_uses_automatic_search_freeze_and_native_replay(tmp_path):
    panel = prices()
    spec = MiningFactorSpec("risk_adjusted_momentum", 5)
    search = auto_mine_factors(panel.head(90), research_direction="momentum", windows=(5,), budget=3)
    assert spec.name in search.trials["candidate"].to_list()
    selection = mine_factors(panel.head(90), [spec])
    assert selection.selected == (spec,)
    receipt = save_mining_result(selection, tmp_path / "selection")
    frozen = load_mining_result(tmp_path / "selection", manifest_sha256=receipt)
    raw = panel.unpivot(index="date", variable_name="symbol", value_name="close").with_columns(
        *(pl.col("close").alias(field) for field in ("open", "high", "low")), pl.lit(1_000_000.).alias("volume"))
    dataset = load_backtest_dataset(bars=raw, factors=raw.select("date", "symbol").with_columns(pl.lit(0.).alias("factor")))
    result = run_mining_execution_pipeline(frozen, dataset, execution_bars=raw)[spec.name]
    assert result.event.fills
    assert min(fill.timestamp for fill in result.event.fills) > frozen.inner_end
