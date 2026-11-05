import numpy as np
import polars as pl
import pytest

from cortexbacktest.research import hierarchical_risk_parity, optimize_weights


def returns():
    return pl.DataFrame({"date": [1, 2, 3, 4], "A": [-.01, -.01, .01, .01],
                         "B": [-.02, .02, -.02, .02]})


def test_hrp_analytical_inverse_variance_and_label_order():
    expected = {"A": .8, "B": .2}
    assert hierarchical_risk_parity(returns()) == pytest.approx(expected)
    assert hierarchical_risk_parity(returns().select("date", "B", "A").reverse()) == pytest.approx(expected)
    assert hierarchical_risk_parity(returns().select("date", "A")) == {"A": 1.}


def test_gmv_caps_turnover_and_constant_expected_returns():
    pytest.importorskip("cvxpy")
    assert optimize_weights(returns()) == pytest.approx({"A": .8, "B": .2}, abs=1e-5)
    assert optimize_weights(returns(), expected_returns={"A": .01, "B": .01}) == pytest.approx({"A": .8, "B": .2}, abs=1e-5)
    prior = {"B": .5, "A": .5}
    limited = optimize_weights(returns(), previous=prior, max_turnover=.2)
    assert limited == pytest.approx({"A": .6, "B": .4}, abs=1e-5)
    assert optimize_weights(returns(), max_weight=.55) == pytest.approx({"A": .55, "B": .45}, abs=1e-5)
    assert optimize_weights(returns(), previous=prior, max_turnover=0) == pytest.approx(prior, abs=1e-7)


def test_infeasible_and_failed_solver_do_not_return_fallback(monkeypatch):
    cp = pytest.importorskip("cvxpy")
    with pytest.raises(ValueError, match="status"):
        optimize_weights(returns(), previous={"A": 1., "B": 0.}, max_weight=.6, max_turnover=.1)
    def fail(*args, **kwargs):
        raise cp.error.SolverError("unavailable")
    monkeypatch.setattr(cp.Problem, "solve", fail)
    with pytest.raises(ValueError, match="solver failed"):
        optimize_weights(returns())


def test_inaccurate_status_is_rejected(monkeypatch):
    cp = pytest.importorskip("cvxpy")
    def inaccurate(problem, **kwargs):
        problem._status = cp.OPTIMAL_INACCURATE
    monkeypatch.setattr(cp.Problem, "solve", inaccurate)
    with pytest.raises(ValueError, match="status"):
        optimize_weights(returns())


@pytest.mark.parametrize("kwargs", [
    {"max_weight": .4}, {"max_weight": float("nan")}, {"risk_aversion": 0},
    {"max_turnover": .2}, {"previous": {"C": .5, "B": .5}},
    {"previous": {"A": -.1, "B": .5}}, {"expected_returns": {"A": 0}},
    {"expected_returns": {"A": float("inf"), "B": 0}},
])
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        optimize_weights(returns(), **kwargs)


@pytest.mark.parametrize("function", [hierarchical_risk_parity, optimize_weights])
@pytest.mark.parametrize("kind", ["null", "inf", "constant", "duplicate", "short"])
def test_invalid_history(function, kind):
    frame = returns()
    if kind == "null":
        frame = frame.with_columns(pl.lit(None, dtype=pl.Float64).alias("A"))
    elif kind == "inf":
        frame = frame.with_columns(pl.lit(np.inf).alias("A"))
    elif kind == "constant":
        frame = frame.with_columns(pl.lit(1.).alias("A"))
    elif kind == "duplicate":
        frame = frame.with_columns(pl.lit(1).alias("date"))
    else:
        frame = frame.head(1)
    with pytest.raises(ValueError):
        function(frame)
