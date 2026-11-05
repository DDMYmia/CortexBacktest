import numpy as np
import pytest

from cortexbacktest.research import (
    WalkForwardSplit,
    benjamini_hochberg,
    deflated_sharpe_ratio,
    purged_walk_forward_splits,
    reality_check,
    run_walk_forward,
    sharpe_haircut,
    stationary_bootstrap_indices,
    student_t_two_sided_p_value,
    superior_predictive_ability,
    t_test_mean_zero,
)


@pytest.mark.parametrize("field", ["train_size", "test_size", "step_size", "embargo"])
@pytest.mark.parametrize("value", [True, 2.5, float("nan"), float("inf")])
def test_walk_forward_rejects_noninteger_session_counts(field, value):
    kwargs = {"train_size": 5, "test_size": 2, field: value}
    with pytest.raises(ValueError, match="integer"):
        WalkForwardSplit(**kwargs)


@pytest.mark.parametrize("horizon", [True, 1.5, float("nan"), float("inf")])
def test_purged_split_rejects_noninteger_label_horizon(horizon):
    with pytest.raises(ValueError, match="integer"):
        purged_walk_forward_splits(range(10), train_size=5, test_size=2, label_horizon=horizon)


def test_walk_forward_split_respects_embargo():
    splits = WalkForwardSplit(4, 2, embargo=1).split(range(10))

    assert [(train.tolist(), test.tolist()) for train, test in splits] == [
        ([0, 1, 2, 3], [5, 6]),
        ([0, 1, 2, 3, 4, 5], [7, 8]),
    ]


def test_purged_split_preserves_numpy_integer_counts():
    splits = purged_walk_forward_splits(range(10), train_size=np.int64(5),
                                       test_size=np.int64(2), label_horizon=np.int64(0),
                                       step_size=np.int64(2), embargo=np.int64(0))
    assert [(train.tolist(), test.tolist()) for train, test in splits] == [
        ([0, 1, 2, 3, 4], [5, 6]), ([0, 1, 2, 3, 4, 5, 6], [7, 8])]


def test_purged_splits_remove_overlapping_forward_labels():
    train, test = purged_walk_forward_splits(range(10), train_size=5, test_size=2, label_horizon=2)[0]

    assert train.tolist() == [0, 1, 2]
    assert test.tolist() == [5, 6]


def test_run_walk_forward_isolates_each_callback_boundary():
    calls = []

    def evaluate(train, test):
        calls.append((train.tolist(), test.tolist()))
        train[:] = -1
        return int(test[0])

    result = run_walk_forward(
        range(8),
        evaluate,
        splitter=WalkForwardSplit(train_size=3, test_size=2),
    )

    assert [item.value for item in result] == [3, 5]
    assert calls == [([0, 1, 2], [3, 4]), ([0, 1, 2, 3, 4], [5, 6])]
    assert result[0].train_indices.tolist() == [0, 1, 2]


def test_deflated_sharpe_penalizes_more_trials():
    one_trial = deflated_sharpe_ratio(0.1, observations=252, trials=1)
    many_trials = deflated_sharpe_ratio(0.1, observations=252, trials=100)

    assert 0 <= many_trials < one_trial <= 1
    assert np.isfinite(many_trials)


@pytest.mark.parametrize(
    # Two-sided critical t values from published Student's t tables.
    ("degrees_of_freedom", "critical_t", "alpha"),
    [(1, 12.706, 0.05), (5, 2.571, 0.05), (20, 2.086, 0.05), (100, 1.984, 0.05), (10, 3.169, 0.01), (30, 2.750, 0.01)],
)
def test_student_t_p_value_matches_published_critical_values(degrees_of_freedom, critical_t, alpha):
    assert student_t_two_sided_p_value(critical_t, degrees_of_freedom) == pytest.approx(alpha, abs=5e-4)


def test_student_t_p_value_is_symmetric_and_one_at_zero():
    assert student_t_two_sided_p_value(0.0, 10) == pytest.approx(1.0)
    assert student_t_two_sided_p_value(2.5, 15) == pytest.approx(student_t_two_sided_p_value(-2.5, 15))


def test_t_test_mean_zero_detects_signal_and_ignores_noise():
    rng = np.random.default_rng(0)
    signal = t_test_mean_zero(rng.normal(0.5, 1.0, 200))
    noise = t_test_mean_zero(rng.normal(0.0, 1.0, 200))
    assert signal.observations == 200
    assert signal.p_value < 1e-6
    assert noise.p_value > 0.05


def test_t_test_mean_zero_handles_degenerate_samples():
    assert np.isnan(t_test_mean_zero([]).p_value)
    assert np.isnan(t_test_mean_zero([1.0]).p_value)

    # Zero variance is a limit, not an error: constant non-zero drives t to
    # +/-inf and p to 0, constant zero leaves the test undefined.
    constant_positive = t_test_mean_zero([2.0, 2.0])
    assert constant_positive.t_stat == np.inf
    assert constant_positive.p_value == 0.0
    assert t_test_mean_zero([-2.0, -2.0]).t_stat == -np.inf
    assert np.isnan(t_test_mean_zero([0.0, 0.0]).t_stat)
    assert np.isnan(t_test_mean_zero([0.0, 0.0]).p_value)


def test_benjamini_hochberg_matches_published_example():
    # Benjamini & Hochberg (1995) worked example: 4 discoveries at alpha=0.05.
    p_values = np.array(
        [0.0001, 0.0004, 0.0019, 0.0095, 0.0201, 0.0278, 0.0298, 0.0344, 0.0459, 0.3240, 0.4262, 0.5719, 0.6528, 0.7590, 1.000]
    )
    result = benjamini_hochberg(p_values, alpha=0.05)
    assert result.discoveries == 4
    assert result.rejected[:4].all()
    assert not result.rejected[4:].any()


def test_benjamini_hochberg_is_less_permissive_than_raw_alpha():
    p_values = np.array([0.01, 0.02, 0.03, 0.04, 0.045])
    result = benjamini_hochberg(p_values, alpha=0.05)
    assert (result.q_values >= p_values).all()
    assert result.discoveries <= int((p_values <= 0.05).sum())


def test_benjamini_hochberg_q_values_are_monotone_and_bounded():
    rng = np.random.default_rng(3)
    p_values = rng.beta(0.5, 3.0, 40)
    result = benjamini_hochberg(p_values, alpha=0.1)
    ordered = result.q_values[np.argsort(p_values)]
    assert np.all(np.diff(ordered) >= -1e-12)
    assert np.all((result.q_values >= 0.0) & (result.q_values <= 1.0))


def test_benjamini_hochberg_carries_nan_through_without_rejecting():
    result = benjamini_hochberg(np.array([0.001, np.nan, 0.5]), alpha=0.05)
    assert np.isnan(result.q_values[1])
    assert not result.rejected[1]
    # family size is 2, not 3, so the smallest q is 0.001 * 2 / 1
    assert result.q_values[0] == pytest.approx(0.002)


def test_benjamini_hochberg_rejects_invalid_input():
    with pytest.raises(ValueError, match="alpha"):
        benjamini_hochberg([0.1], alpha=0.0)
    with pytest.raises(ValueError, match="p_values must be in"):
        benjamini_hochberg([0.5, 1.5])


def _family(seed: int, edge: float = 0.0, columns: int = 20, periods: int = 500) -> np.ndarray:
    """A searched family: `columns` strategies, only column 0 carrying `edge`."""
    matrix = np.random.default_rng(seed).normal(0.0, 1.0, (periods, columns))
    matrix[:, 0] += edge
    return matrix


def test_stationary_bootstrap_preserves_runs_and_is_reproducible():
    first = stationary_bootstrap_indices(50, resamples=8, block_size=5, seed=11)
    again = stationary_bootstrap_indices(50, resamples=8, block_size=5, seed=11)

    assert first.shape == (8, 50)
    assert np.array_equal(first, again)
    assert first.min() >= 0 and first.max() < 50
    # Consecutive draws continue the previous index far more often than chance
    # (1/50) would give: that continuation is the serial dependence the block
    # bootstrap exists to keep.
    continued = np.mean(first[:, 1:] == (first[:, :-1] + 1) % 50)
    assert continued > 0.6
    iid = stationary_bootstrap_indices(50, resamples=8, block_size=1, seed=11)
    assert np.mean(iid[:, 1:] == (iid[:, :-1] + 1) % 50) < 0.2

    with pytest.raises(ValueError):
        stationary_bootstrap_indices(50, resamples=8, block_size=0.5, seed=11)


def test_reality_check_prices_in_the_size_of_the_search():
    noise = reality_check(_family(0), resamples=400, seed=1)
    assert noise.p_value > 0.2                       # the best of 20 noise columns is luck
    assert 0 <= noise.best < 20

    real = reality_check(_family(0, edge=0.25), resamples=400, seed=1)
    assert real.best == 0
    assert real.p_value < 0.05
    assert real.statistic > noise.statistic


def test_reality_check_rejects_a_family_that_is_not_a_matrix():
    with pytest.raises(ValueError):
        reality_check(np.zeros(10), resamples=10)
    with pytest.raises(ValueError):
        reality_check(np.full((10, 2), np.nan), resamples=10)


def test_spa_keeps_its_power_when_the_family_is_padded_with_junk():
    """Hansen's improvement over White, as the reason to prefer it.

    Adding 200 hopeless-but-noisy candidates to a family that contains one real
    edge destroys the reality check — the bootstrap maximum is now driven by the
    junk — while SPA studentises and recentres them away.
    """
    real = _family(3, edge=0.16, columns=5, periods=600)
    junk = np.random.default_rng(4).normal(-1.0, 4.0, (600, 200))
    padded = np.hstack([real, junk])

    assert reality_check(real, resamples=400, seed=2).p_value < 0.05
    assert superior_predictive_ability(real, resamples=400, seed=2).p_value < 0.05
    assert reality_check(padded, resamples=400, seed=2).p_value > 0.5
    padded_spa = superior_predictive_ability(padded, resamples=400, seed=2)
    assert padded_spa.p_value < 0.05
    assert padded_spa.best == 0


def test_spa_reports_its_recentring_bounds_in_order():
    result = superior_predictive_ability(_family(5, edge=0.1), resamples=400, seed=6)

    assert result.p_lower is not None and result.p_upper is not None
    assert result.p_lower <= result.p_value <= result.p_upper
    assert result.statistic >= 0.0


def test_spa_refuses_a_family_with_no_variation():
    with pytest.raises(ValueError, match="constant"):
        superior_predictive_ability(np.ones((20, 3)), resamples=10)


def test_sharpe_haircut_grows_with_the_number_of_trials():
    single = sharpe_haircut(0.2, observations=500, trials=1)
    searched = sharpe_haircut(0.2, observations=500, trials=200)

    assert single.haircut == pytest.approx(0.0, abs=1e-9)
    assert single.haircut_sharpe == pytest.approx(0.2)
    assert 0.25 < searched.haircut < 0.4
    assert searched.adjusted_p_value == pytest.approx(min(1.0, single.p_value * 200))
    assert searched.haircut_sharpe < single.haircut_sharpe
    assert sharpe_haircut(0.2, observations=500, trials=1000).haircut > searched.haircut


def test_bhy_is_the_stronger_adjustment_and_holm_matches_bonferroni():
    bonferroni = sharpe_haircut(0.2, observations=500, trials=100, method="bonferroni")
    holm = sharpe_haircut(0.2, observations=500, trials=100, method="holm")
    bhy = sharpe_haircut(0.2, observations=500, trials=100, method="bhy")

    assert holm.haircut_sharpe == pytest.approx(bonferroni.haircut_sharpe)
    assert bhy.haircut_sharpe < bonferroni.haircut_sharpe        # extra harmonic factor
    assert bhy.adjusted_p_value > bonferroni.adjusted_p_value


def test_a_marginal_sharpe_does_not_survive_a_wide_search():
    marginal = sharpe_haircut(0.1, observations=500, trials=200)

    assert marginal.adjusted_p_value == 1.0
    assert marginal.haircut_sharpe == 0.0
    assert marginal.haircut == 1.0


def test_sharpe_haircut_rejects_inputs_it_cannot_price():
    with pytest.raises(ValueError, match="positive reported Sharpe"):
        sharpe_haircut(-0.1, observations=500, trials=10)
    with pytest.raises(ValueError, match="method"):
        sharpe_haircut(0.2, observations=500, trials=10, method="bonferoni")
    with pytest.raises(ValueError):
        sharpe_haircut(0.2, observations=1, trials=10)
