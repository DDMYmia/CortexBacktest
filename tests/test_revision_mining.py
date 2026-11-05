from datetime import datetime, timedelta, timezone

import polars as pl
import pytest

from cortexbacktest.research import MiningFactorSpec, RevisionPanel, WalkForwardSplit, auto_mine_factors, evaluate_holdout, mine_factors
from cortexbacktest.pipelines.mining_store import load_mining_result, save_mining_result
from polars.testing import assert_frame_equal


@pytest.fixture
def panel():
    dates = [datetime(2024, 1, 1, 21, tzinfo=timezone.utc) + timedelta(days=i) for i in range(120)]
    rows = [{"date": day, "symbol": symbol, "price": 100 * (1 + rate) ** i, "recorded_at": day, "known_at": day}
            for symbol, rate in zip("ABCD", (.001, .002, .003, .004), strict=True) for i, day in enumerate(dates)]
    return RevisionPanel(pl.DataFrame(rows), pl.DataFrame({"date": dates, "decision_time": dates}))


def test_shared_leaf_cache_is_local_to_each_study(panel, monkeypatch):
    calls = []
    original = RevisionPanel.factor

    def counted(self, spec):
        calls.append(spec.name)
        return original(self, spec)

    monkeypatch.setattr(RevisionPanel, "factor", counted)
    specs = [MiningFactorSpec("momentum", 5), MiningFactorSpec("add(rank(momentum),rank(sma_ratio))", 5)]
    first = mine_factors(panel.head(90), specs)
    assert calls.count("momentum:5:1") == 1
    assert calls.count("sma_ratio:5:1") == 1
    second = mine_factors(panel.head(90), specs)
    assert calls.count("momentum:5:1") == 2
    assert first.trials.equals(second.trials)


def test_study_period_keeps_received_warmup_but_excludes_it_from_scores(panel, tmp_path):
    from cortexbacktest.research import run_factor_research

    start = panel.decisions[30, "date"]
    # Historical bars arrive together at the research start, not at old dates.
    received = RevisionPanel(panel.revisions.with_columns(
        pl.max_horizontal("recorded_at", pl.lit(start)).alias("recorded_at")), panel.decisions)
    spec = MiningFactorSpec("momentum", 20)
    result = mine_factors(received.head(90), [spec], study_start=start)
    factor = received.head(90).factor(spec)
    assert factor.head(30).select(pl.exclude("date").is_null().all()).row(0) == (True,) * 4
    assert factor[30, "A"] is not None  # Warm-up is available at this decision.
    reference = run_factor_research(received.head(90).prices().slice(30), factor.slice(30),
                                    horizons=(1,), transaction_cost_bps=10.)
    fold = result.trials[0, "folds"].to_list()[0]
    assert fold["start"] == start
    assert fold["observations"] == 59
    assert fold["final_equity"] == reference.final_equity
    assert result.selected == (spec,)
    receipt = save_mining_result(result, tmp_path / "study")
    frozen = load_mining_result(tmp_path / "study", manifest_sha256=receipt)
    assert frozen.settings["study_start"] == start.isoformat()
    assert evaluate_holdout(frozen, received)
    # Warm-up remains bound into the receipt even though it is not scored.
    changed = RevisionPanel(received.revisions.with_columns(
        pl.when(pl.col("date") < start).then(pl.col("price") * 2).otherwise(pl.col("price")).alias("price")),
        received.decisions)
    with pytest.raises(ValueError, match="Inner revision"):
        evaluate_holdout(frozen, changed)


def test_study_start_moves_auto_search_folds_without_discarding_history(panel):
    start = panel.decisions[30, "date"]
    result = auto_mine_factors(panel.head(90), study_start=start, research_direction="momentum",
                              windows=(20,), budget=2, validation=WalkForwardSplit(10, 25))
    for folds in result.trials["folds"]:
        assert [fold["start"] for fold in folds] == [panel.decisions[40, "date"], panel.decisions[65, "date"]]
        assert [fold["observations"] for fold in folds] == [24, 24]
    assert result.inner_end == panel.decisions[89, "date"]
    with pytest.raises(ValueError, match="Inner validation"):
        auto_mine_factors(panel.head(90), study_start=panel.decisions[85, "date"],
                         research_direction="momentum", validation=WalkForwardSplit(10, 25))


@pytest.mark.parametrize("start", [datetime(2024, 2, 1), datetime(2023, 1, 1, tzinfo=timezone.utc),
                                    datetime(2025, 1, 1, tzinfo=timezone.utc), "2024-02-01"])
def test_invalid_study_period_is_rejected(panel, start):
    with pytest.raises(ValueError, match="Study start"):
        mine_factors(panel.head(90), [MiningFactorSpec("momentum", 2)], study_start=start)


@pytest.mark.parametrize("formula", ["add(rank(momentum),rank(rolling_volatility))",
                                      "div(momentum,rolling_volatility)", "rank(sma_ratio)"])
def test_column_composition_equals_per_decision_reference_with_invalidations(panel, formula):
    spec = MiningFactorSpec(formula, 5, -1)
    revisions = panel.revisions.with_columns(pl.lit(None, dtype=pl.String).alias("invalid_reason"))
    invalid = revisions.filter((pl.col("symbol") == "D") & (pl.col("date") == panel.decisions[20, "date"])).with_columns(
        pl.lit(None, dtype=pl.Float64).alias("price"), pl.lit("unverified_price_basis").alias("invalid_reason"),
        pl.lit(panel.decisions[22, "date"]).alias("recorded_at"))
    changed = RevisionPanel(pl.concat([revisions, invalid]), panel.decisions)
    reference = []
    for index, decision in enumerate(changed.decisions.iter_rows(named=True)):
        start = max(0, index - spec.window)
        grid = changed.decisions.slice(start, index - start + 1).select("date")
        reference.append(spec.build(changed.prices(cutoff=decision["decision_time"], grid=grid)).tail(1))
    assert_frame_equal(changed.factor(spec), pl.concat(reference), check_exact=True)


def test_revision_selection_freezes_and_accepts_only_later_new_evidence(panel, tmp_path):
    selected = mine_factors(panel.head(90), [MiningFactorSpec("momentum", 2)])
    assert selected.selected
    receipt = save_mining_result(selected, tmp_path / "selection")
    frozen = load_mining_result(tmp_path / "selection", manifest_sha256=receipt)
    assert frozen.inner_revision_sha256 == selected.inner_revision_sha256
    before = evaluate_holdout(frozen, panel)
    correction = panel.revisions.filter((pl.col("symbol") == "D") & (pl.col("date") == panel.decisions[89, "date"])).with_columns(
        pl.col("price") * 2, pl.lit(panel.decisions[91, "date"]).alias("recorded_at"))
    later = RevisionPanel(pl.concat([panel.revisions, correction]), panel.decisions)
    after = evaluate_holdout(frozen, later)
    name = frozen.selected[0].name
    assert before[name].target_weights.head(1).equals(after[name].target_weights.head(1))
    assert not before[name].target_weights.equals(after[name].target_weights)
    assert frozen.trials.equals(selected.trials)
    changed = RevisionPanel(panel.revisions.with_columns(pl.col("price") * 2), panel.decisions)
    with pytest.raises(ValueError, match="Inner revision"):
        evaluate_holdout(frozen, changed)
    with pytest.raises(ValueError, match="Inner revision"):
        evaluate_holdout(frozen, panel.prices())


def test_old_fold_labels_ignore_revisions_received_in_later_folds(panel):
    validation = WalkForwardSplit(30, 30)
    spec = MiningFactorSpec("momentum", 2)
    baseline = mine_factors(panel.head(90), [spec], validation=validation)
    correction = panel.revisions.filter((pl.col("symbol") == "D") & (pl.col("date") == panel.decisions[40, "date"])).with_columns(
        pl.col("price") * 10, pl.lit(panel.decisions[80, "date"]).alias("recorded_at"))
    revised = mine_factors(RevisionPanel(pl.concat([panel.revisions, correction]), panel.decisions).head(90),
                          [spec], validation=validation)
    assert revised.trials.equals(baseline.trials)
    assert revised.inner_revision_sha256 != baseline.inner_revision_sha256
    assert revised.inner_sha256 != baseline.inner_sha256


def test_revision_input_order_is_irrelevant_and_auto_search_reuses_same_engine(panel):
    shuffled = RevisionPanel(panel.revisions.reverse(), panel.decisions.reverse())
    assert panel.head(90).fingerprint() == shuffled.head(90).fingerprint()
    result = auto_mine_factors(panel.head(90), research_direction="momentum", windows=(2, 5), budget=2)
    assert result.selected and result.trials.height == 2
    assert evaluate_holdout(result, shuffled)


def test_delayed_decisions_are_not_scored_as_if_they_traded_at_close(panel):
    late = RevisionPanel(panel.revisions, panel.decisions.with_columns(pl.col("decision_time") + timedelta(minutes=1)))
    with pytest.raises(ValueError, match="execution-aligned labels"):
        mine_factors(late, [MiningFactorSpec("momentum", 2)])


def test_revision_selection_runs_through_same_event_pipeline(panel):
    from cortexbacktest import load_backtest_dataset
    from cortexbacktest.pipelines import run_mining_execution_pipeline

    selected = mine_factors(panel.head(90), [MiningFactorSpec("momentum", 2)])
    raw = panel.prices().unpivot(index="date", variable_name="symbol", value_name="close").with_columns(
        *(pl.col("close").alias(c) for c in ("open", "high", "low")), pl.lit(1_000_000.).alias("volume"))
    dataset = load_backtest_dataset(bars=raw, factors=raw.select("date", "symbol").with_columns(pl.lit(0.).alias("factor")))
    result = next(iter(run_mining_execution_pipeline(selected, dataset, execution_bars=raw, revision_panel=panel).values()))
    assert result.event.fills
    assert min(fill.timestamp for fill in result.event.fills) == panel.decisions[91, "date"]
    with pytest.raises(ValueError, match="match the revision evaluation snapshot"):
        run_mining_execution_pipeline(selected, dataset, execution_bars=raw, revision_panel=panel.head(119))


def executable_panels(panel, *, opening_lag=timedelta(hours=6, minutes=30)):
    cutoffs = panel.decisions.with_columns(pl.col("decision_time") + timedelta(minutes=30))
    features = RevisionPanel(panel.revisions.with_columns(pl.col("recorded_at") + timedelta(minutes=15)), cutoffs)
    prints = panel.revisions.with_columns(
        pl.col("date") - opening_lag, pl.col("recorded_at") - opening_lag,
        pl.col("known_at") - opening_lag,
    )
    execution = RevisionPanel(prints, cutoffs.with_columns(pl.col("date") - opening_lag))
    return features, execution


def test_delayed_automatic_mining_freezes_execution_evidence_and_evaluates_later(panel, tmp_path):
    features, execution = executable_panels(panel)
    selected = auto_mine_factors(features.head(90), execution=execution.head(90),
                                research_direction="momentum", windows=(2, 5), budget=2,
                                validation=WalkForwardSplit(30, 30))
    assert selected.selected
    assert selected.inner_execution_sha256 == execution.head(90).fingerprint()
    assert selected.settings["return_basis"] == "execution_prints"
    receipt = save_mining_result(selected, tmp_path / "delayed")
    frozen = load_mining_result(tmp_path / "delayed", manifest_sha256=receipt)
    assert frozen.inner_execution_sha256 == selected.inner_execution_sha256
    reports = evaluate_holdout(frozen, features, execution=execution)
    assert reports
    for report in reports.values():
        assert report.target_weights["date"].min() > features.decisions[89, "decision_time"]
        assert report.final_equity > 1
    changed = RevisionPanel(execution.revisions.with_columns(pl.col("price") * 2), execution.decisions)
    with pytest.raises(ValueError, match="Inner execution"):
        evaluate_holdout(frozen, features, execution=changed)


def test_later_execution_revisions_cannot_improve_earlier_folds(panel):
    features, execution = executable_panels(panel)
    spec = MiningFactorSpec("momentum", 2)
    cfg = {"validation": WalkForwardSplit(30, 30)}
    baseline = mine_factors(features.head(90), [spec], execution=execution.head(90), **cfg)
    correction = execution.revisions.filter((pl.col("symbol") == "D") & (
        pl.col("date") == execution.decisions[40, "date"])).with_columns(
            pl.col("price") * 10, pl.lit(features.decisions[80, "decision_time"]).alias("recorded_at"))
    revised = RevisionPanel(pl.concat([execution.revisions, correction]), execution.decisions)
    after = mine_factors(features.head(90), [spec], execution=revised.head(90), **cfg)
    assert after.trials.equals(baseline.trials)
    assert after.inner_execution_sha256 != baseline.inner_execution_sha256
    assert after.inner_sha256 == baseline.inner_sha256
    correction = correction.with_columns(pl.lit(features.decisions[91, "decision_time"]).alias("recorded_at"))
    later = RevisionPanel(pl.concat([execution.revisions, correction]), execution.decisions)
    assert evaluate_holdout(baseline, features, execution=later)


def test_crypto_holdout_skips_opening_before_selection_was_frozen(panel):
    features, execution = executable_panels(panel, opening_lag=timedelta(days=1))
    selected = mine_factors(features.head(90), [MiningFactorSpec("momentum", 2)], execution=execution.head(90))
    report = next(iter(evaluate_holdout(selected, features, execution=execution).values()))
    assert execution.decisions[90, "date"] < features.decisions[89, "decision_time"]
    assert report.target_weights[0, "date"] == execution.decisions[91, "date"]


def test_unreceived_execution_prices_do_not_pass_mining_gate(panel):
    features, execution = executable_panels(panel)
    late = RevisionPanel(execution.revisions.with_columns(pl.col("recorded_at") + timedelta(days=365)), execution.decisions)
    result = mine_factors(features.head(90), [MiningFactorSpec("momentum", 2)], execution=late.head(90))
    assert not result.selected
    assert result.trials[0, "status"] == "insufficient_evidence"
    with pytest.raises(ValueError, match="same per-session"):
        mine_factors(features, [MiningFactorSpec("momentum", 2)], execution=execution.head(119))


@pytest.mark.parametrize("opening_lag,first", [(timedelta(hours=6), 90), (timedelta(days=1), 91)])
def test_delayed_mining_runs_native_event_pipeline_without_extra_signal_lag(panel, opening_lag, first):
    from cortexbacktest import load_backtest_dataset
    from cortexbacktest.pipelines import run_mining_execution_pipeline

    features, execution = executable_panels(panel, opening_lag=opening_lag)
    selected = mine_factors(features.head(90), [MiningFactorSpec("momentum", 2)], execution=execution.head(90))
    raw = features.prices().unpivot(index="date", variable_name="symbol", value_name="close").with_columns(
        *(pl.col("close").alias(c) for c in ("open", "high", "low")), pl.lit(1_000_000.).alias("volume"))
    dataset = load_backtest_dataset(bars=raw, factors=raw.select("date", "symbol").with_columns(pl.lit(0.).alias("factor")))
    result = next(iter(run_mining_execution_pipeline(selected, dataset, execution_bars=raw,
                        revision_panel=features, execution=execution, explain_execution=True).values()))
    assert min(fill.timestamp for fill in result.event.fills) == execution.decisions[first, "date"]
    assert result.research.target_weights[0, "date"] == execution.decisions[first, "date"]
    assert result.attribution.to_dict()["reconciliation_residual"] == pytest.approx(0.)
    changed = raw.with_columns(pl.col("open") * .99, pl.col("low") * .99)
    with pytest.raises(ValueError, match="Raw opens must match"):
        run_mining_execution_pipeline(selected, dataset, execution_bars=changed,
                                      revision_panel=features, execution=execution)


def test_frozen_targets_keep_revision_evidence_and_accept_latest_decision(panel):
    from cortexbacktest.research import frozen_factor_targets

    result = mine_factors(panel.head(90), [MiningFactorSpec("momentum", 5)])
    name = result.selected[0].name
    latest = frozen_factor_targets(result, panel.head(91))[name]
    assert latest.height == 1
    assert_frame_equal(latest, frozen_factor_targets(result, panel)[name].head(1))
    changed = RevisionPanel(panel.revisions.with_columns(pl.col("known_at") + timedelta(seconds=1)), panel.decisions)
    with pytest.raises(ValueError, match="Inner revision"):
        frozen_factor_targets(result, changed)


def test_frozen_factor_sizes_current_holdings_without_future_returns(panel, tmp_path):
    from cortexbacktest.event import RebalanceConfig
    from cortexbacktest.pipelines import frozen_factor_quantity_changes

    result = mine_factors(panel.head(90), [MiningFactorSpec("momentum", 5)])
    checkpoint = tmp_path / "selection"
    digest = save_mining_result(result, checkpoint)
    now = panel.decisions[90, "decision_time"]
    args = dict(manifest_sha256=digest, factor=result.selected[0].name, revision_panel=panel.head(91),
                as_of=now, prices=dict.fromkeys("ABCD", 100.), quantities={"A": 0., "B": 2., "C": 0., "D": 10.},
                equity=10_000., rebalance=RebalanceConfig(cash_buffer=.1))
    # The 20% tails are short A / long D. B is a previously held asset to exit.
    assert frozen_factor_quantity_changes(checkpoint, **args) == {"A": -90., "B": -2., "D": 80.}
    fractional_holdings = {"A": .25, "B": 2., "C": 0., "D": 10.75}
    # Whole-lot policy truncates the trade delta, not the final holding.
    assert frozen_factor_quantity_changes(checkpoint, **{**args, "quantities": fractional_holdings}) == {
        "A": -90., "B": -2., "D": 79.}
    assert frozen_factor_quantity_changes(checkpoint, **{
        **args, "quantities": fractional_holdings,
        "rebalance": RebalanceConfig(cash_buffer=.1, allow_fractional=True, min_trade_value=250.),
    }) == {"A": -90.25, "D": 79.25}
    # Missing current features are not zero targets, even with valid raw prices.
    # Both partial and full data outages must preserve the distinction.
    for missing in (["D"], list("ABCD")):
        incomplete = RevisionPanel(panel.revisions.filter(
            ~((pl.col("date") == now) & pl.col("symbol").is_in(missing))), panel.decisions.head(91))
        with pytest.raises(ValueError, match="Incomplete factor decision"):
            frozen_factor_quantity_changes(checkpoint, **{**args, "revision_panel": incomplete})
    # Only one post-selection close exists: no next-day label or opening price.
    for update, error in [
        ({"quantities": {"A": 0.}}, "exactly cover"),
        ({"prices": {**args["prices"], "D": float("nan")}}, "finite"),
        ({"equity": float("inf")}, "finite"),
        ({"factor": "not-selected"}, "frozen selection"),
        ({"as_of": now - timedelta(seconds=1)}, "cutoff"),
        ({"as_of": now.replace(tzinfo=None)}, "aware"),
    ]:
        with pytest.raises(ValueError, match=error):
            frozen_factor_quantity_changes(checkpoint, **{**args, **update})


def test_delayed_frozen_targets_require_only_training_execution_evidence(panel):
    from cortexbacktest.research import frozen_factor_targets

    features, execution = executable_panels(panel)
    result = mine_factors(features.head(90), [MiningFactorSpec("momentum", 2)], execution=execution.head(90))
    assert result.selected
    targets = frozen_factor_targets(result, features.head(91), inner_execution=execution.head(90))
    first = targets[result.selected[0].name]
    assert first.height == 1
    assert first[0, "date"] == features.decisions[90, "decision_time"]
    assert first[0, "date"] > features.decisions[90, "date"]
    # No opening print or return label from the new decision's future was supplied.
    assert_frame_equal(first, frozen_factor_targets(result, features, inner_execution=execution.head(90))[
        result.selected[0].name].head(1))
    with pytest.raises(ValueError, match="Inner execution"):
        frozen_factor_targets(result, features.head(91))


@pytest.mark.parametrize("pinned", [False, True])
def test_frozen_decisions_run_in_existing_simulator_without_holdout_labels(panel, tmp_path, pinned):
    from cortexbacktest import load_backtest_dataset
    from cortexbacktest.pipelines import BrokerExecutionConfig, run_frozen_factor_simulation

    if pinned:
        panel = RevisionPanel(panel.revisions.with_columns(pl.col("symbol").replace_strict(
            {s: f"INSTRUMENT:{i}" for i, s in enumerate("ABCD", 1)})), panel.decisions)
    features, execution = executable_panels(panel)
    result = mine_factors(features.head(90), [MiningFactorSpec("momentum", 2), MiningFactorSpec("momentum", 5)], execution=execution.head(90))
    digest = save_mining_result(result, tmp_path / "frozen")
    # Two later sessions suffice for a decision followed by its first executable open.
    recent = features.head(92)
    raw = recent.prices().unpivot(index="date", variable_name="symbol", value_name="close").with_columns(
        *(pl.col("close").alias(c) for c in ("open", "high", "low")), pl.lit(1_000_000.).alias("volume"))
    dataset = load_backtest_dataset(bars=raw, factors=raw.select("date", "symbol").with_columns(pl.lit(0.).alias("factor")))
    openings = dict(zip(recent.decisions["date"], execution.head(92).prices()["date"], strict=True))
    inputs = dict(dataset=dataset, revision_panel=recent, execution_bars=raw, inner_execution=execution.head(90),
                  broker=BrokerExecutionConfig(opening_times=openings, commission_bps=10))
    if pinned:
        from cortexbacktest.pipelines.factor_strategy import FrozenFactorStrategy

        diagnostics = run_frozen_factor_simulation(tmp_path / "frozen", manifest_sha256=digest,
                                                   trial_family=True, **inputs)
        assert set(diagnostics) == set(result.trials["candidate"])
        discarded = set(diagnostics) - {spec.name for spec in result.selected}
        assert discarded
        for name in discarded:
            assert diagnostics[name].run_manifest["metadata"]["mode"] == "frozen-trial-diagnostic"
            assert "strategyArtifact" not in diagnostics[name].run_manifest["metadata"]
            with pytest.raises(ValueError, match="not part of the frozen selection"):
                FrozenFactorStrategy.from_checkpoint(tmp_path / "frozen", selection_sha256=digest,
                    factor=name, symbols=tuple(sorted(set(raw["symbol"]))))
        with pytest.raises(ValueError, match="cannot replace a pinned strategy"):
            run_frozen_factor_simulation(tmp_path / "frozen", manifest_sha256=digest,
                                        trial_family=True, factor=result.selected[0].name, **inputs)
    if pinned:
        from cortexbacktest.pipelines.factor_strategy import FactorValuation, FrozenFactorStrategy
        from cortexbacktest.shadow import PortfolioObservation
        from decimal import Decimal
        from hashlib import sha256
        from dataclasses import replace

        strategy = FrozenFactorStrategy.from_checkpoint(tmp_path / "frozen", selection_sha256=digest,
            factor=result.selected[0].name, symbols=tuple(sorted(set(raw["symbol"]))))
        strategy = FrozenFactorStrategy.from_bytes(strategy.to_bytes(), expected_sha256=strategy.sha256)
        assert replace(strategy, rebalance=replace(strategy.rebalance, cash_buffer=.1)).sha256 != strategy.sha256
        invalid = strategy.to_bytes() + b" "
        with pytest.raises(ValueError, match="canonical"):
            FrozenFactorStrategy.from_bytes(invalid, expected_sha256=sha256(invalid).hexdigest())
        event = strategy.simulate(tmp_path / "frozen", **inputs)
        assert event.run_manifest["metadata"]["strategySha256"] == strategy.sha256
        # Provenance is derived from the enforced object, not an arbitrary hash
        # supplied beside unrelated simulation parameters.
        for conflict in ({"factor": "different"},
                         {"rebalance": replace(strategy.rebalance, cash_buffer=.1)},
                         {"manifest_sha256": "a" * 64}):
            with pytest.raises(ValueError, match="conflict with frozen strategy"):
                run_frozen_factor_simulation(tmp_path / "frozen", **{
                    **inputs, "manifest_sha256": digest, "strategy": strategy, **conflict})
        prices = {s: 100. for s in strategy.symbols}
        observed_at = recent.decisions[90, "decision_time"]
        portfolio = PortfolioObservation("diagnostic-account", "d" * 64, observed_at,
                                         tuple((s, Decimal(0)) for s in strategy.symbols), ())
        valuation = FactorValuation(portfolio, "e" * 64, observed_at, "USD", Decimal(10_000),
                                    tuple((s, Decimal(100)) for s in prices))
        changes = strategy.quantity_changes(tmp_path / "frozen", revision_panel=recent.head(91),
            as_of=observed_at, valuation=valuation, max_age_seconds=60, max_signal_age_seconds=60, inner_execution=execution.head(90))
        assert changes == {"INSTRUMENT:1": -100., "INSTRUMENT:4": 100.}
        assert strategy.research_only is True
        wrong_clock = RevisionPanel(recent.head(91).revisions, recent.head(91).decisions,
                                    knowledge_policy='source_publication_replay')
        with pytest.raises(ValueError, match='Inner revision evidence'):
            strategy.quantity_changes(tmp_path / "frozen", revision_panel=wrong_clock,
                as_of=observed_at, valuation=valuation, max_age_seconds=60,
                max_signal_age_seconds=60, inner_execution=execution.head(90))
        from cortexbacktest.reporting.shadow_delivery import write_shadow_decision
        import json

        delivery_valuation = replace(valuation, source_sha256=portfolio.source_sha256)
        decision = strategy.shadow_decision(tmp_path / "frozen", revision_panel=recent.head(91),
            as_of=observed_at, valuation=delivery_valuation, max_age_seconds=60,
            max_signal_age_seconds=60, inner_execution=execution.head(90))
        receipt = write_shadow_decision(decision, tmp_path / "factor-decision.json")
        assert json.loads((tmp_path / "factor-decision.json").read_bytes()) == decision
        assert receipt["decisionSha256"] == decision["decisionSha256"]
        assert decision["executionAuthorized"] is False
        assert decision["accountSourceSha256"] == portfolio.source_sha256
        assert decision["inputSha256"] == recent.head(91).fingerprint()
        # Retain shorts and HOLDs; this diagnostic does not turn Spot-ineligible
        # targets into a different strategy by clipping or dropping rows.
        rows = {row["symbol"]: row for row in decision["decisions"]}
        assert len(rows) == len(strategy.symbols)
        assert rows["INSTRUMENT:1"]["targetQuantity"] == "-100.0"
        assert rows["INSTRUMENT:4"]["quantity"] == "100.0"
        assert rows["INSTRUMENT:2"]["side"] == "HOLD"
        with pytest.raises(ValueError, match="one holdings and valuation receipt"):
            strategy.shadow_decision(tmp_path / "frozen", revision_panel=recent.head(91),
                as_of=observed_at, valuation=valuation, max_age_seconds=60,
                max_signal_age_seconds=60, inner_execution=execution.head(90))
        # Fresh account marks cannot renew an expired research signal.
        later = observed_at + timedelta(seconds=60)
        fresh = replace(valuation, event_time=later, portfolio=replace(portfolio, event_time=later))
        with pytest.raises(ValueError, match="Stale or future factor"):
            strategy.quantity_changes(tmp_path / "frozen", revision_panel=recent.head(91),
                as_of=later, valuation=fresh, max_age_seconds=60, max_signal_age_seconds=60,
                inner_execution=execution.head(90))
        for limit in (0, -1, True, 86401):
            with pytest.raises(ValueError, match="bounded freshness"):
                strategy.quantity_changes(tmp_path / "frozen", revision_panel=recent.head(91),
                    as_of=observed_at, valuation=valuation, max_age_seconds=60,
                    max_signal_age_seconds=limit, inner_execution=execution.head(90))
        with pytest.raises(ValueError, match="scope"):
            strategy.quantity_changes(tmp_path / "frozen", revision_panel=recent.head(91), as_of=observed_at,
                                      valuation=replace(valuation, prices=(("A", Decimal(100)),)), max_age_seconds=60, max_signal_age_seconds=60)
        for invalid in (
            replace(valuation, event_time=observed_at - timedelta(seconds=60)),
            replace(valuation, event_time=observed_at + timedelta(seconds=1)),
            replace(valuation, portfolio=replace(portfolio, event_time=observed_at - timedelta(seconds=60))),
            replace(valuation, portfolio=replace(portfolio, pending_symbols=(strategy.symbols[0],))),
            replace(valuation, equity=Decimal("1.123456789012345678")),
            replace(valuation, equity=Decimal("NaN")),
            replace(valuation, source_sha256="missing"),
            replace(valuation, currency="mixed"),
        ):
            with pytest.raises(ValueError):
                strategy.quantity_changes(tmp_path / "frozen", revision_panel=recent.head(91), as_of=observed_at,
                    valuation=invalid, max_age_seconds=60, max_signal_age_seconds=60, inner_execution=execution.head(90))
    else:
        event = run_frozen_factor_simulation(str(tmp_path / "frozen"), manifest_sha256=digest,
                                             **inputs)[result.selected[0].name]
    assert event.run_manifest["metadata"]["selectionSha256"] == digest
    assert event.run_manifest["metadata"]["revisionSha256"] == recent.fingerprint()
    for bars, same in ((raw.reverse(), True), (raw.with_columns(pl.col("volume") * 2), False)):
        changed = {**inputs, "execution_bars": bars}
        replay = (strategy.simulate(tmp_path / "frozen", **changed) if pinned else
                  run_frozen_factor_simulation(str(tmp_path / "frozen"), manifest_sha256=digest,
                                               **changed)[result.selected[0].name])
        assert (replay.run_manifest["data_sha256"] == event.run_manifest["data_sha256"]) is same
        assert (replay.run_manifest == event.run_manifest) is same
    assert event.fills
    assert all(fill.timestamp > recent.decisions[90, "decision_time"] for fill in event.fills)
    assert min(fill.timestamp for fill in event.fills) == openings[recent.decisions[91, "date"]]
    assert event.summary()["final_equity"] < 100_000  # costs on flat open/close execution bars
    curve = event.performance.equity_curve()
    cutoff = features.decisions[89, "decision_time"]
    expected_sessions = [day for day in recent.decisions["date"] if openings[day] > cutoff]
    assert curve["timestamp"].to_list() == expected_sessions
    from cortexbacktest.contracts import Side

    expected_cash = 100_000 - sum(
        (1 if fill.side == Side.BUY else -1) * fill.quantity * fill.price + fill.commission
        for fill in event.fills)
    assert event.performance.equity_curve()[-1, "cash"] == pytest.approx(expected_cash)
    for position in event.positions:
        quantity = sum((1 if fill.side == Side.BUY else -1) * fill.quantity
                       for fill in event.fills if fill.symbol == position.symbol)
        assert position.quantity == pytest.approx(quantity)
