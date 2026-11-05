from dataclasses import replace

import numpy as np
import polars as pl
import pytest

from cortexbacktest.pipelines.factor_workflow import FactorWorkflowConfig, evaluate_return_family


def returns():
    rng = np.random.default_rng(7)
    return pl.DataFrame({"date": list(range(120)), "signal": rng.normal(.02, .01, 120),
                         "weak": rng.normal(-.01, .01, 120)})


def test_omitted_candidate_prevents_support_even_when_supplied_winner_passes():
    config = FactorWorkflowConfig(resamples=100, trials=2)
    complete = evaluate_return_family(returns(), config=config)
    assert complete["family"]["complete"] is True
    assert complete["statistically_supported"] == ("signal",)
    missing = evaluate_return_family(returns(), config=replace(config, trials=3))
    assert missing["family"]["complete"] is False
    assert missing["statistically_supported"] == ()


def test_annualization_does_not_change_return_family_verdict():
    config = FactorWorkflowConfig(resamples=100, trials=2)
    assert evaluate_return_family(returns(), config=config) == evaluate_return_family(
        returns().reverse(), config=replace(config, periods_per_year=365))


def test_search_count_must_not_be_inferred_from_only_supplied_winners():
    with pytest.raises(ValueError, match="explicit full search"):
        evaluate_return_family(returns(), config=FactorWorkflowConfig(resamples=100))


def test_constant_cash_returns_remain_explicitly_undefined_and_unsupported():
    verdict = evaluate_return_family(returns().with_columns(pl.lit(0.).alias("signal"), pl.lit(0.).alias("weak")),
                                     config=FactorWorkflowConfig(resamples=100, trials=2))
    assert verdict["family"]["status"] == "undefined_constant_returns"
    assert verdict["statistically_supported"] == ()


@pytest.mark.parametrize("broken", [
    lambda r: r.with_columns(pl.lit(1).alias("date")),
    lambda r: r.with_columns(pl.lit(float("nan")).alias("signal")),
    lambda r: r.with_columns(pl.lit(float("inf")).alias("signal")),
    lambda r: r.with_columns(pl.lit(None, dtype=pl.Float64).alias("signal")),
    lambda r: r.head(1),
])
def test_incomplete_observations_are_not_dropped_to_make_a_verdict(broken):
    with pytest.raises(ValueError):
        evaluate_return_family(broken(returns()), config=FactorWorkflowConfig(resamples=100, trials=2))
