from dataclasses import replace
from datetime import datetime, timedelta, timezone

import polars as pl
import pytest

from cortexbacktest.research import (
    MiningFactorSpec, RevisionPanel, WalkForwardSplit,
    evaluate_holdout, frozen_factor_values, mine_factors,
)
from cortexbacktest.pipelines.mining_store import save_mining_result, load_mining_result
from cortexbacktest.pipelines.mining_execution import frozen_factor_quantity_changes
from cortexbacktest.pipelines.factor_strategy import FrozenFactorStrategy


def inputs():
    dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(100)]
    prices = pl.DataFrame({"date": dates, **{s: [100 * (1 + r) ** i for i in range(100)]
                                           for s, r in zip("ABCD", (.001, .002, .003, .004), strict=True)}})
    return prices, pl.DataFrame({"date": dates, **{s: dates for s in "ABCD"}})


@pytest.mark.parametrize("indicator", ["momentum", "sma_ratio", "rolling_zscore", "rolling_volatility"])
def test_late_history_can_warm_later_signals_without_backdating(indicator):
    prices, availability = inputs()
    availability = availability.with_columns(
        pl.when(pl.col("date") == prices[10, "date"]).then(pl.lit(prices[12, "date"]))
        .otherwise(pl.col("A")).alias("A"))
    spec = MiningFactorSpec(indicator, 5)
    actual = spec.build(prices, availability)
    assert actual[10, "A"] is None
    later = 15 if indicator == "momentum" else 12
    assert actual[later, "A"] == pytest.approx(spec.build(prices)[later, "A"])
    permanently_masked = prices.with_columns(pl.when(pl.col("date") == prices[10, "date"])
                                             .then(None).otherwise(pl.col("A")).alias("A"))
    assert spec.build(permanently_masked)[later, "A"] is None


def test_late_label_cannot_change_earlier_fold_feedback():
    prices, availability = inputs()
    availability = availability.with_columns(
        pl.when(pl.col("date") == prices[45, "date"]).then(pl.lit(prices[55, "date"]))
        .otherwise(pl.col("A")).alias("A"))
    changed = prices.with_columns(pl.when(pl.col("date") == prices[45, "date"])
                                  .then(pl.col("A") * 20).otherwise(pl.col("A")).alias("A"))
    kwargs = dict(availability=availability, validation=WalkForwardSplit(20, 30))
    a = mine_factors(prices, [MiningFactorSpec("momentum", 2)], **kwargs)
    b = mine_factors(changed, [MiningFactorSpec("momentum", 2)], **kwargs)
    assert a.trials["folds"].to_list()[0][0] == b.trials["folds"].to_list()[0][0]
    assert a.trials["folds"].to_list()[0][0]["final_equity"] is None
    assert not a.selected


def test_checkpoint_preserves_receipts_and_holdout_requires_same_inner_receipts(tmp_path):
    prices, availability = inputs()
    frozen = mine_factors(prices.head(70), [MiningFactorSpec("momentum", 2)], availability=availability.head(70))
    receipt = save_mining_result(frozen, tmp_path / "selection")
    restored = load_mining_result(tmp_path / "selection", manifest_sha256=receipt)
    assert restored.inner_availability_sha256 == frozen.inner_availability_sha256
    assert evaluate_holdout(restored, prices, availability=availability)
    with pytest.raises(ValueError, match="availability changed"):
        evaluate_holdout(restored, prices)
    with pytest.raises(ValueError, match="availability changed"):
        evaluate_holdout(restored, prices, availability=availability.with_columns(pl.col("A") + pl.duration(days=1)))


def test_misaligned_receipt_panel_is_rejected():
    prices, availability = inputs()
    with pytest.raises(ValueError, match="same dates"):
        mine_factors(prices, [MiningFactorSpec("momentum", 2)], availability=availability.head(99))
    with pytest.raises(ValueError, match="Datetime"):
        mine_factors(prices, [MiningFactorSpec("momentum", 2)], availability=availability.with_columns(pl.lit(0).alias("A")))


def test_frozen_replay_bridge_uses_received_clocks_and_only_latest_factor(tmp_path, monkeypatch):
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    anchor = start + timedelta(days=99, hours=1)
    rows = []
    for i in range(111):
        for asset in range(1, 5):
            date = start + timedelta(days=i)
            rows.append({"date": date, "symbol": f"INSTRUMENT:{asset}", "price": 100 * (1 + asset / 1000) ** i,
                         "known_at": date, "recorded_at": anchor if i < 100 else date + timedelta(minutes=2)})
    revisions = pl.DataFrame(rows)
    decisions = revisions.select("date").unique().sort("date").with_columns(
        pl.when(pl.col("date") < start + timedelta(days=100)).then(pl.col("date"))
        .otherwise(pl.col("date") + pl.duration(minutes=3)).alias("decision_time"))
    history = RevisionPanel(revisions.filter(pl.col("date") < start + timedelta(days=100)),
                            decisions.head(100), "source_publication_replay")
    selection = mine_factors(history.head(70), [MiningFactorSpec("momentum", 2)])
    assert selection.selected
    selection = replace(selection, settings={**selection.settings, "knowledge_policy": "source_publication_replay",
                                             "replay_as_of": anchor.isoformat()})
    digest = save_mining_result(selection, tmp_path / "selection")
    panel = RevisionPanel(revisions, decisions, "source_publication_replay", anchor)
    spec = selection.selected[0]
    assert frozen_factor_values(selection, panel)[spec.name].tail(1).equals(
        frozen_factor_values(selection, panel, factor=spec.name, last_only=True)[spec.name])
    calls = []
    original = RevisionPanel.factor
    def observed(self, spec, **kwargs):
        calls.append((spec.name, kwargs))
        return original(self, spec, **kwargs)
    monkeypatch.setattr(RevisionPanel, "factor", observed)
    changes = frozen_factor_quantity_changes(str(tmp_path / "selection"), manifest_sha256=digest, factor=spec.name,
        revision_panel=panel, as_of=decisions[-1, "decision_time"],
        prices={f"INSTRUMENT:{i}": 100. for i in range(1, 5)}, quantities={f"INSTRUMENT:{i}": 0. for i in range(1, 5)}, equity=10_000.)
    assert changes and calls == [(spec.name, {"last_only": True})]
    strategy = FrozenFactorStrategy.from_checkpoint(tmp_path / "selection", selection_sha256=digest,
        factor=spec.name, symbols=tuple(f"INSTRUMENT:{i}" for i in range(1, 5)))
    assert strategy.research_only and b'"researchOnly":true' in strategy.to_bytes()
    late = revisions.with_columns(pl.when(pl.col("date") == decisions[-1, "date"])
        .then(pl.col("recorded_at") + pl.duration(minutes=5)).otherwise(pl.col("recorded_at")).alias("recorded_at"))
    delayed = RevisionPanel(late, decisions, "source_publication_replay", anchor)
    with pytest.raises(ValueError, match="Incomplete factor decision"):
        frozen_factor_quantity_changes(str(tmp_path / "selection"), manifest_sha256=digest, factor=spec.name,
            revision_panel=delayed, as_of=decisions[-1, "decision_time"], prices={}, quantities={}, equity=10_000.)
    calls.clear()
    holdings = {f"INSTRUMENT:{i}": 0. for i in range(1, 5)}
    kwargs = dict(manifest_sha256=digest, factor=spec.name, revision_panel=panel,
                  as_of=decisions[-1, "decision_time"], equity=0.,
                  prices={symbol: 100. for symbol in holdings}, quantities=holdings)
    assert frozen_factor_quantity_changes(str(tmp_path / "selection"), **kwargs) == {}
    assert calls == []  # Verified zero capital cannot benefit from recomputing factors.
    with pytest.raises(ValueError, match="Zero capital requires explicit zero holdings"):
        frozen_factor_quantity_changes(str(tmp_path / "selection"), **{**kwargs, "quantities": {**holdings, "INSTRUMENT:1": 1.}})
