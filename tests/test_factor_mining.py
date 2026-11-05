from dataclasses import replace
from datetime import datetime, timedelta

import polars as pl
import pytest

from cortexbacktest.research.mining import MiningFactorSpec as FactorSpec
from cortexbacktest.research.mining import auto_mine_factors, evaluate_holdout, mine_factors


def prices():
    return pl.DataFrame({
        "date": [datetime(2025, 1, 1) + timedelta(days=i) for i in range(100)],
        **{name: [100 * (1 + rate) ** i for i in range(100)]
           for name, rate in zip("ABCD", (0.001, 0.002, 0.003, 0.004), strict=True)},
    })


def test_mining_does_not_replace_existing_composite_factor_spec():
    from cortexbacktest.research import FactorSpec as CompositeSpec, MiningFactorSpec
    from cortexbacktest.research.factors import FactorSpec as ExistingSpec

    assert CompositeSpec is ExistingSpec
    assert MiningFactorSpec is FactorSpec


def test_selection_logs_duplicates_and_rejects_redundant_signals():
    candidate = FactorSpec("momentum", 2)
    result = mine_factors(prices().head(70), [candidate, candidate, FactorSpec("momentum", 3)])
    assert result.selected == (candidate,)
    assert result.trials["status"].to_list() == ["selected", "duplicate", "redundant"]
    assert result.trials[0, "rank_ic"] == pytest.approx(1)
    assert result.trials[2, "max_correlation"] == pytest.approx(1)


def test_holdout_does_not_reselect_and_has_no_inner_dates():
    panel = prices()
    result = mine_factors(panel.head(70), [FactorSpec("momentum", 2)])
    before = result.trials.clone()
    reports = evaluate_holdout(result, panel)
    report = reports[result.selected[0].name]
    assert report.target_weights_frame()["date"].min() > result.inner_end
    assert report.horizon_results[1].rank_ic_by_date.height <= 30
    # Altering the outer period affects evaluation, never the frozen trial ledger.
    changed = panel.with_columns(pl.when(pl.col("date") > result.inner_end)
                                 .then(pl.col("A") * 2).otherwise(pl.col("A")).alias("A"))
    evaluate_holdout(result, changed)
    assert result.trials.equals(before)
    assert result.selected == (FactorSpec("momentum", 2),)
    with pytest.raises(ValueError, match="Inner data changed"):
        evaluate_holdout(result, panel.with_columns(pl.col("A") * 2))
    with pytest.raises(ValueError, match="Insufficient later"):
        evaluate_holdout(result, panel.head(70))


@pytest.mark.parametrize("indicator", ["momentum", "rolling_zscore", "sma_ratio", "rolling_volatility"])
def test_builders_are_causal(indicator):
    spec = FactorSpec(indicator, 5)
    panel = prices()
    assert spec.build(panel).head(70).equals(spec.build(panel.head(70)))


def test_weak_and_short_trials_are_not_promoted():
    assert not mine_factors(prices(), [FactorSpec("momentum", 2, -1)]).selected
    result = mine_factors(prices().head(10), [FactorSpec("momentum", 2)])
    assert result.trials[0, "status"] == "insufficient_evidence"
    assert not result.selected


def test_positive_ic_does_not_override_cost_or_turnover_gates():
    candidate = FactorSpec("momentum", 2)
    feedback = []
    costly = mine_factors(prices(), [candidate], cost_bps=5000,
                         on_evaluated=lambda spec, score: feedback.append(score))
    assert costly.trials[0, "rank_ic"] == pytest.approx(1.)
    assert costly.trials[0, "net_return"] < 0
    assert costly.trials[0, "status"] == "cost_rejected"
    assert not costly.selected and feedback == [None]
    constrained = mine_factors(prices(), [candidate], max_turnover=0)
    assert constrained.trials[0, "net_return"] > 0
    assert constrained.trials[0, "turnover"] > 0
    assert constrained.trials[0, "status"] == "turnover_rejected"
    assert not constrained.selected
    assert constrained.settings["max_turnover"] == 0
    automatic = auto_mine_factors(prices(), research_direction="momentum", budget=2,
                                  cost_bps=5000, windows=(2,))
    assert not automatic.selected
    assert set(automatic.trials["status"]) == {"cost_rejected"}


@pytest.mark.parametrize("kwargs", [{"min_net_return": -1}, {"min_net_return": float("nan")},
                                   {"max_turnover": -1}, {"max_turnover": float("inf")}])
def test_cost_gate_thresholds_are_validated(kwargs):
    with pytest.raises(ValueError, match="thresholds"):
        mine_factors(prices(), [FactorSpec("momentum", 2)], **kwargs)


@pytest.mark.parametrize("change", [
    {"indicator": "python:eval"}, {"window": -1}, {"direction": 0}, {"window": True},
])
def test_invalid_spec_refused(change):
    with pytest.raises(ValueError):
        replace(FactorSpec("momentum", 2), **change)


def test_panel_and_threshold_errors_are_explicit():
    spec = FactorSpec("momentum", 2)
    with pytest.raises(ValueError, match="three assets"):
        mine_factors(prices().select("date", "A"), [spec])
    with pytest.raises(ValueError, match="unique"):
        mine_factors(pl.concat([prices(), prices().head(1)]), [spec])
    with pytest.raises(ValueError, match="finite"):
        mine_factors(prices().with_columns(pl.lit(float("inf")).alias("A")), [spec])
    with pytest.raises(ValueError, match="thresholds"):
        mine_factors(prices(), [spec], max_correlation=float("nan"))


def test_automatic_search_is_bounded_deterministic_and_uses_feedback(monkeypatch):
    from cortexbacktest.research import mining

    original = mining.SurrogateOptimizer.suggest
    history_lengths = []

    def suggest(self, grid, observations, **kwargs):
        history_lengths.append(len(observations))
        return original(self, grid, observations, **kwargs)

    monkeypatch.setattr(mining.SurrogateOptimizer, "suggest", suggest)
    a = auto_mine_factors(prices().head(70), research_direction="momentum", budget=5)
    b = auto_mine_factors(prices().head(70), research_direction="momentum", budget=5)
    assert history_lengths == [0, 1, 2, 3, 4] * 2
    assert a.trials.height == a.trials["candidate"].n_unique() == 5
    assert a.trials.equals(b.trials)
    assert a.settings["research_direction"] == "momentum"
    assert all(s.direction == 1 for s in a.selected)
    assert a.inner_sha256 == b.inner_sha256


def test_automatic_direction_is_not_changed_to_fit_returns():
    result = auto_mine_factors(prices(), research_direction="mean_reversion", windows=(2,), budget=20)
    assert result.trials.height == 6  # three leaves plus three distinct ranked pairs
    assert all(name.endswith(":-1") for name in result.trials["candidate"])


@pytest.mark.parametrize("settings", [
    {"research_direction": "anything"}, {"budget": 0}, {"budget": True},
    {"windows": ()}, {"windows": (5, 5)}, {"windows": (-1,)},
])
def test_invalid_automatic_request_is_rejected(settings):
    with pytest.raises(ValueError):
        auto_mine_factors(prices(), **{"research_direction": "momentum", **settings})


def test_inner_folds_clip_labels_keep_warmup_and_feed_only_fold_scores():
    from cortexbacktest.research import WalkForwardSplit

    panel = prices()
    spec = FactorSpec("momentum", 5)
    feedback = []
    result = mine_factors(panel, [spec], horizon=2,
                          validation=WalkForwardSplit(20, 40),
                          on_evaluated=lambda candidate, score: feedback.append(score))
    folds = result.trials["folds"].to_list()[0]
    assert [f["observations"] for f in folds] == [38, 38]
    assert [f["start"] for f in folds] == [panel[20, "date"], panel[60, "date"]]
    assert feedback == pytest.approx([1.0])
    assert result.trials[0, "net_return"] == pytest.approx(folds[0]["final_equity"] * folds[1]["final_equity"] - 1)
    assert result.trials[0, "turnover"] == pytest.approx((folds[0]["turnover"] + folds[1]["turnover"]) / 2)
    assert result.settings["validation"]["train_size"] == 20
    # Altering every price in the later fold cannot affect earlier labels.
    changed = panel.with_columns(pl.when(pl.col("date") >= panel[60, "date"])
                                 .then(10000 / pl.col("A")).otherwise(pl.col("A")).alias("A"))
    other = mine_factors(changed, [spec], horizon=2, validation=WalkForwardSplit(20, 40))
    assert other.trials["folds"].to_list()[0][0] == folds[0]


def test_missing_inner_fold_cannot_be_hidden_by_other_fold():
    from cortexbacktest.research import WalkForwardSplit

    panel = prices().with_columns(pl.when(pl.col("date") >= prices()[60, "date"])
                                  .then(None).otherwise(pl.col(name)).alias(name) for name in "ABCD")
    feedback = []
    result = mine_factors(panel, [FactorSpec("momentum", 2)],
                          validation=WalkForwardSplit(20, 40),
                          on_evaluated=lambda candidate, score: feedback.append(score))
    assert not result.selected
    assert result.trials[0, "status"] == "insufficient_evidence"
    assert feedback == [None]


@pytest.mark.parametrize("train,test,step", [(100, 30, 30), (20, 30, 10), (20, 1, 1)])
def test_invalid_inner_fold_layout_is_rejected(train, test, step):
    from cortexbacktest.research import WalkForwardSplit

    with pytest.raises(ValueError, match="Inner validation"):
        auto_mine_factors(prices(), research_direction="momentum",
                          validation=WalkForwardSplit(train, test, step))


def test_automatic_inner_fold_selection_then_frozen_outer_backtest():
    from cortexbacktest.research import WalkForwardSplit

    panel = prices()
    result = auto_mine_factors(panel.head(75), research_direction="momentum",
                               windows=(2, 5), budget=4,
                               validation=WalkForwardSplit(25, 25))
    assert result.selected
    assert all(len(folds) == 2 for folds in result.trials["folds"].to_list())
    reports = evaluate_holdout(result, panel)
    assert set(reports) == {spec.name for spec in result.selected}
    assert all(report.equity_curve_frame()["date"].min() > result.inner_end for report in reports.values())


def test_frozen_targets_need_no_future_labels_and_match_holdout(tmp_path):
    from cortexbacktest.pipelines.mining_store import load_mining_result, save_mining_result
    from cortexbacktest.research import frozen_factor_targets

    panel = prices()
    selected = mine_factors(panel.head(70), [FactorSpec("momentum", 2)])
    digest = save_mining_result(selected, tmp_path / "selection")
    frozen = load_mining_result(tmp_path / "selection", manifest_sha256=digest)
    name = frozen.selected[0].name
    # One new observation is enough for a decision, but cannot supply a forward return.
    latest = frozen_factor_targets(frozen, panel.head(71))[name]
    assert latest.height == 1
    with pytest.raises(ValueError, match="Insufficient later"):
        evaluate_holdout(frozen, panel.head(71))
    full = frozen_factor_targets(frozen, panel)[name]
    assert latest.equals(full.head(1))
    assert full.equals(evaluate_holdout(frozen, panel)[name].target_weights_frame())
    changed = panel.with_columns(pl.when(pl.col("date") > panel[70, "date"])
                                 .then(pl.col("A") * 10).otherwise(pl.col("A")).alias("A"))
    assert latest.equals(frozen_factor_targets(frozen, changed)[name].head(1))
    with pytest.raises(ValueError, match="Inner data changed"):
        frozen_factor_targets(frozen, panel.with_columns(pl.col("A") * 2))
