"""The factor family pipeline, and the ways a result can flatter itself."""

from __future__ import annotations

import json

import numpy as np
import polars as pl
import pytest

from cortexbacktest.pipelines.factor_workflow import (
    FactorWorkflowConfig, run_factor_workflow,
)


def panel(*, sessions: int = 260, names: int = 200, loading: float = 0.0,
          noise: float = 0.05, seed: int = 0, junk: int = 1) -> pl.DataFrame:
    """A synthetic cross-section where `signal` predicts with a known strength."""
    rng = np.random.default_rng(seed)
    parts = []
    for day in range(sessions):
        factor = rng.normal(size=names)
        columns = {
            "date": [day] * names,
            "symbol": [f"s{i}" for i in range(names)],
            "signal": factor,
            "forward_return": loading * factor + rng.normal(size=names) * noise,
        }
        for index in range(junk):
            columns[f"junk{index}"] = rng.normal(size=names)
        parts.append(pl.DataFrame(columns))
    return pl.concat(parts)


CONFIG = FactorWorkflowConfig(min_names=50, resamples=400, seed=3)


class TestWhatItFinds:
    def test_a_planted_signal_is_established(self) -> None:
        result = run_factor_workflow(panel(loading=0.02), ["signal", "junk0"], config=CONFIG)
        assert result.established == ("signal",)
        assert result.family["best"] == "signal"

    def test_pure_noise_establishes_nothing(self) -> None:
        result = run_factor_workflow(panel(loading=0.0, seed=11),
                                     ["signal", "junk0"], config=CONFIG)
        assert result.established == ()

    def test_spa_rejection_blocks_establishment(self, monkeypatch) -> None:
        from types import SimpleNamespace
        import cortexbacktest.pipelines.factor_workflow as workflow

        monkeypatch.setattr(workflow, "superior_predictive_ability",
                            lambda *a, **k: SimpleNamespace(p_value=0.9))
        result = run_factor_workflow(panel(loading=0.02), ["signal"], config=CONFIG)
        assert result.family["spa_p"] == 0.9
        assert result.established == ()


class TestWhatItRefusesToHide:
    def test_a_factor_whose_ic_and_spread_disagree_is_flagged(self) -> None:
        """Momentum's own shape: monotone almost always, and the extreme winners
        crash occasionally. Rare large losses barely move a rank correlation and
        dominate a mean spread, so the IC says buy while the long-short
        portfolio loses. Ranking a family by IC buys the losing side of this.
        """
        rng = np.random.default_rng(7)
        names, parts = 300, []
        for day in range(520):
            factor = rng.normal(size=names)
            uniform = (factor.argsort().argsort() + 0.5) / names
            forward = 0.02 * (uniform - 0.5)
            if rng.random() < 0.05:                      # one session in twenty
                forward = forward - 0.60 * (uniform > 0.8)
            parts.append(pl.DataFrame({
                "date": [day] * names, "symbol": [f"s{i}" for i in range(names)],
                "crashy": factor,
                "forward_return": forward + rng.normal(size=names) * 0.01}))
        result = run_factor_workflow(pl.concat(parts), ["crashy"], config=CONFIG)
        one = result.evaluations[0]
        assert one.rank_ic > 0 > one.spread_sharpe
        assert one.sign_disagreement
        assert result.established == ()

    def test_turnover_and_breakeven_cost_are_not_optional(self) -> None:
        result = run_factor_workflow(panel(loading=0.02), ["signal"], config=CONFIG)
        one = result.evaluations[0]
        assert 0 <= one.turnover <= 1
        assert one.breakeven_cost_bps > 0
        # There is no constructor that yields an evaluation without them.
        assert set(("turnover", "breakeven_cost_bps", "net_spread_annual")) <= set(
            result.report()["factors"][0])

    def test_costs_can_take_a_factor_away(self) -> None:
        expensive = FactorWorkflowConfig(min_names=50, resamples=200, seed=3, cost_bps=500.0)
        result = run_factor_workflow(panel(loading=0.02), ["signal"], config=expensive)
        assert not result.evaluations[0].survives_costs
        assert result.established == ()

    def test_one_factor_is_still_a_search(self) -> None:
        # Evaluating a single candidate does not exempt it from being priced as
        # the outcome of a search; it is a family of one, not a free hypothesis.
        result = run_factor_workflow(panel(loading=0.02), ["signal"], config=CONFIG)
        assert result.family["trials"] == 1
        assert result.family["deflated_sharpe"] is not None

    def test_ic_autocorrelation_is_measured_not_assumed(self) -> None:
        result = run_factor_workflow(panel(loading=0.02), ["signal"], config=CONFIG)
        assert -1 <= result.evaluations[0].ic_ar1 <= 1


class TestTheRecord:
    def test_the_manifest_identifies_the_data_and_the_settings(self) -> None:
        frame = panel(loading=0.02)
        first = run_factor_workflow(frame, ["signal"], config=CONFIG, data_id="fixture")
        again = run_factor_workflow(frame, ["signal"], config=CONFIG, data_id="fixture")
        assert first.manifest["data_digest"] == again.manifest["data_digest"]
        assert first.manifest["data_id"] == "fixture"
        assert first.manifest["config"]["cost_bps"] == CONFIG.cost_bps
        json.dumps(first.report())

    def test_a_changed_panel_changes_the_digest(self) -> None:
        a = run_factor_workflow(panel(loading=0.02), ["signal"], config=CONFIG)
        b = run_factor_workflow(panel(loading=0.03), ["signal"], config=CONFIG)
        assert a.manifest["data_digest"] != b.manifest["data_digest"]


class TestInputsItRefuses:
    def test_a_missing_column_is_named(self) -> None:
        with pytest.raises(ValueError, match="missing columns"):
            run_factor_workflow(panel(), ["nope"], config=CONFIG)

    def test_an_empty_family_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least one factor"):
            run_factor_workflow(panel(), [], config=CONFIG)

    def test_a_cross_section_too_thin_to_be_one_is_refused(self) -> None:
        with pytest.raises(ValueError, match="too thin"):
            run_factor_workflow(panel(names=20), ["signal"],
                                config=FactorWorkflowConfig(min_names=200))

    def test_an_all_null_factor_is_refused_rather_than_scored(self) -> None:
        frame = panel().with_columns(pl.lit(None, pl.Float64).alias("empty"))
        with pytest.raises(ValueError, match="null on every row"):
            run_factor_workflow(frame, ["empty"], config=CONFIG)

    @pytest.mark.parametrize("bad", [{"quantiles": 1}, {"cost_bps": -1.0},
                                     {"alpha": 0.0}, {"min_names": 2}])
    def test_an_unusable_configuration_is_refused(self, bad: dict) -> None:
        with pytest.raises(ValueError):
            FactorWorkflowConfig(**bad)


def test_sparse_factors_use_common_dates_and_recompute_statistics() -> None:
    frame = panel(sessions=30, names=60, loading=0.02).with_columns(
        pl.when(pl.col("date") != 2).then(pl.col("signal")).alias("signal"),
        pl.when(pl.col("date") != 27).then(pl.col("junk0")).alias("junk0"),
    )
    result = run_factor_workflow(frame, ["signal", "junk0"], config=CONFIG)
    expected = run_factor_workflow(frame.filter(~pl.col("date").is_in([2, 27])),
                                   ["signal", "junk0"], config=CONFIG)
    assert result.evaluations == expected.evaluations
    assert result.family == expected.family
    assert result.daily["date"].n_unique() == 28
    assert result.manifest["coverage"]["signal"] == {
        "raw_sessions": 30, "valid_sessions": 29, "excluded_unresolved_labels": 0, "common_sessions": 28}


def test_min_names_is_checked_after_nonfinite_and_null_filtering() -> None:
    frame = panel(sessions=20, names=60).with_columns(
        pl.when(pl.col("symbol").is_in([f"s{i}" for i in range(15)]))
        .then(float("nan")).otherwise(pl.col("signal")).alias("signal"))
    with pytest.raises(ValueError, match="too thin"):
        run_factor_workflow(frame, ["signal"], config=CONFIG)


def test_duplicate_identity_is_rejected() -> None:
    frame = panel(sessions=10, names=60)
    with pytest.raises(ValueError, match="duplicate date/symbol"):
        run_factor_workflow(pl.concat([frame, frame.head(1)]), ["signal"], config=CONFIG)


def test_custom_columns_and_row_permutations_preserve_results() -> None:
    frame = panel(sessions=20, names=60)
    result = run_factor_workflow(frame, ["signal"], config=CONFIG)
    renamed = frame.reverse().rename({"date": "day", "symbol": "ticker", "forward_return": "target"})
    other = run_factor_workflow(renamed, ["signal"], config=CONFIG,
                                date_col="day", symbol_col="ticker", forward_return_col="target")
    assert other.evaluations == result.evaluations
    assert other.manifest["data_digest"] == result.manifest["data_digest"]


def test_costs_include_short_leg_and_initial_entry() -> None:
    from cortexbacktest.pipelines.factor_workflow import _daily_turnover

    # Long remains C; short changes A -> B. Initial gross entry is 2; the
    # subsequent short cover plus replacement is another 2.
    frame = pl.DataFrame({"date": [0, 0, 0, 1, 1, 1],
                          "symbol": ["A", "B", "C"] * 2,
                          "factor": [1, 2, 3, 2, 1, 3]})
    assert _daily_turnover(frame, 3)["turnover"].to_list() == [0.5, 0.5]


def test_each_candidate_needs_its_own_bh_rejection_and_white_gate(monkeypatch):
    from types import SimpleNamespace
    import cortexbacktest.pipelines.factor_workflow as workflow

    frame = panel(loading=0.02).with_columns(pl.col("signal").alias("copy"))
    monkeypatch.setattr(workflow, "benjamini_hochberg",
                        lambda *a, **k: SimpleNamespace(rejected=np.array([True, False])))
    accepted = run_factor_workflow(frame, ["signal", "copy"], config=CONFIG)
    assert accepted.established == ("signal",)
    monkeypatch.setattr(workflow, "reality_check", lambda *a, **k: SimpleNamespace(p_value=0.9))
    assert run_factor_workflow(frame, ["signal", "copy"], config=CONFIG).established == ()


def test_candidate_cannot_inherit_winners_sharpe_corrections(monkeypatch):
    from types import SimpleNamespace
    import cortexbacktest.pipelines.factor_workflow as workflow

    frame = panel(loading=0.02).with_columns(pl.col("signal").alias("copy"))
    # Identical planted signals isolate the correction gate from the data.
    dsr = iter([0.99, 0.5])
    monkeypatch.setattr(workflow, "deflated_sharpe_ratio", lambda *a, **k: next(dsr))
    monkeypatch.setattr(workflow, "sharpe_haircut",
                        lambda *a, **k: SimpleNamespace(adjusted_p_value=0.001))
    result = run_factor_workflow(frame, ["signal", "copy"], config=CONFIG)
    assert result.established == ("signal",)
    assert result.report()["corrections"]["copy"]["deflated_sharpe"] == 0.5


def test_explicit_search_count_reaches_every_sharpe_correction(monkeypatch):
    from dataclasses import replace
    from types import SimpleNamespace
    import cortexbacktest.pipelines.factor_workflow as workflow

    counts = []
    def deflated(*args, **kwargs):
        counts.append(kwargs["trials"])
        return 0.99
    def haircut(*args, **kwargs):
        counts.append(kwargs["trials"])
        return SimpleNamespace(adjusted_p_value=0.001)
    monkeypatch.setattr(workflow, "deflated_sharpe_ratio", deflated)
    monkeypatch.setattr(workflow, "sharpe_haircut", haircut)
    result = run_factor_workflow(panel(loading=0.02), ["signal"],
                                 config=replace(CONFIG, trials=37))
    assert result.family["trials"] == 37
    assert result.family["complete"] is False
    assert result.established == ()
    assert counts == [37, 37]
    with pytest.raises(ValueError, match="smaller"):
        run_factor_workflow(panel(), ["signal", "junk0"], config=replace(CONFIG, trials=1))


def test_bh_threshold_counts_omitted_search_trials(monkeypatch):
    from dataclasses import replace
    import cortexbacktest.pipelines.factor_workflow as workflow

    actual = workflow.benjamini_hochberg
    thresholds = []
    def check(values, *, alpha):
        thresholds.append(alpha)
        return actual(values, alpha=alpha)
    monkeypatch.setattr(workflow, "benjamini_hochberg", check)
    run_factor_workflow(panel(loading=0.02), ["signal"], config=replace(CONFIG, trials=20))
    assert thresholds == [CONFIG.alpha / 20]


@pytest.mark.parametrize("missing", [None, float("nan"), float("inf")])
def test_missing_label_cannot_select_future_survivors(missing):
    frame = panel(sessions=30, names=60).with_columns(
        pl.when((pl.col("date") == 10) & (pl.col("symbol") == "s0"))
        .then(pl.lit(missing, dtype=pl.Float64)).otherwise(pl.col("forward_return"))
        .alias("forward_return"))
    with pytest.raises(ValueError, match="resolve label coverage and terminal outcomes"):
        run_factor_workflow(frame, ["signal"], config=CONFIG)


def test_excluding_unresolved_labels_is_a_stated_choice_that_is_counted():
    from dataclasses import replace

    frame = panel(sessions=30, names=60).with_columns(
        pl.when((pl.col("date") == 10) & (pl.col("symbol").is_in(["s0", "s1"])))
        .then(None).otherwise(pl.col("forward_return")).alias("forward_return"))
    result = run_factor_workflow(frame, ["signal"], config=replace(CONFIG, unresolved_labels="exclude"))
    assert result.manifest["config"]["unresolved_labels"] == "exclude"
    assert result.manifest["coverage"]["signal"]["excluded_unresolved_labels"] == 2
    with pytest.raises(ValueError, match="unresolved_labels"):
        replace(CONFIG, unresolved_labels="ignore")


def test_unavailable_factor_is_not_a_missing_eligible_label():
    frame = panel(sessions=30, names=60).with_columns(
        pl.when(pl.col("symbol") != "s0").then(pl.col("signal")).alias("signal"),
        pl.when(pl.col("symbol") != "s0").then(pl.col("forward_return")).alias("forward_return"))
    assert run_factor_workflow(frame, ["signal"], config=CONFIG).evaluations[0].sessions == 30


@pytest.mark.parametrize("frequency", [12, 52])
def test_annualization_frequency_changes_units_without_changing_period_returns_or_verdict(frequency):
    from dataclasses import replace

    frame = panel(sessions=40, names=60, loading=0.01)
    config = replace(CONFIG, resamples=50)
    daily = run_factor_workflow(frame, ["signal", "junk0"], config=config)
    result = run_factor_workflow(frame, ["signal", "junk0"],
                                 config=replace(config, periods_per_year=frequency))
    assert result.daily.equals(daily.daily)
    assert result.established == daily.established
    assert result.manifest["config"]["periods_per_year"] == frequency
    assert result.manifest["data_digest"] == daily.manifest["data_digest"]
    for old, new in zip(daily.evaluations, result.evaluations, strict=True):
        assert new.spread_annual == pytest.approx(old.spread_annual * frequency / 252)
        assert new.net_spread_annual == pytest.approx(old.net_spread_annual * frequency / 252)
        assert new.spread_sharpe == pytest.approx(old.spread_sharpe * np.sqrt(frequency / 252))
        assert new.ic_ir == pytest.approx(old.ic_ir * np.sqrt(frequency / 252))
        assert new.p_value == old.p_value
        assert new.turnover == old.turnover
        assert new.breakeven_cost_bps == old.breakeven_cost_bps
        assert result.corrections[new.name] == pytest.approx(daily.corrections[old.name])


@pytest.mark.parametrize("frequency", [0, -12, float("nan"), float("inf"), True])
def test_invalid_annualization_frequency_is_rejected(frequency):
    with pytest.raises(ValueError, match="periods_per_year"):
        FactorWorkflowConfig(periods_per_year=frequency)
