from datetime import datetime, timedelta, timezone

import numpy as np
import polars as pl
import pytest

from cortexbacktest.adapters.qlib_predictions import _constrained_targets

# Every allocation here solves with cvxpy (the optional "optimizer" extra).
pytest.importorskip("cvxpy")


def panel():
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    changes = np.tile([[-0.01, -0.02], [-0.01, 0.02], [0.01, -0.02], [0.01, 0.02]], (20, 1))
    values = 100 * np.vstack([np.ones(2), np.cumprod(1 + changes, axis=0)])
    latest = {
        (str(j + 1), str(i)): {
            "sessionClose": (start + timedelta(days=i)).isoformat(),
            "instrumentId": str(j + 1),
            "close": float(price),
        }
        for i, row in enumerate(values)
        for j, price in enumerate(row)
    }
    targets = pl.DataFrame(
        {
            "date": [start + timedelta(days=i) for i in range(20, 81)],
            "INSTRUMENT:1": [1.0] * 61,
            "INSTRUMENT:2": [0.0] * 61,
        }
    )
    return latest, targets


def allocate(latest, targets):
    return _constrained_targets(
        targets,
        latest,
        20,
        method="gmv",
        max_weight=0.9,
        max_turnover=0.1,
        risk_aversion=None,
    )


def test_gmv_closed_form_uses_all_frozen_assets_and_preserves_budget():
    latest, targets = panel()
    out, turnover = allocate(latest, targets)
    assert out["INSTRUMENT:1"].to_numpy() == pytest.approx(np.full(61, 0.8), abs=1e-5)
    assert out["INSTRUMENT:2"].to_numpy() == pytest.approx(np.full(61, 0.2), abs=1e-5)
    assert turnover[0]["l1"] == pytest.approx(1.0) and not turnover[0]["budgetApplied"]
    assert all(t["budgetApplied"] and t["l1"] <= 0.1 + 1e-7 for t in turnover[1:])


def test_mvo_trailing_mean_closed_form_and_binding_budget():
    latest, targets = panel()
    # Shift the second half of the history; the optimum must change but budget binds.
    for (instrument, i), row in latest.items():
        if instrument == "2" and int(i) > 30:
            row["close"] *= 1.01 ** (int(i) - 30)
    out, turnover = _constrained_targets(
        targets, latest, 20, method="mvo", max_weight=0.9, max_turnover=0.04, risk_aversion=10.0
    )
    assert out["INSTRUMENT:1"][0] == pytest.approx(0.8, abs=1e-5)
    assert any(t["l1"] > 0.0399 for t in turnover[1:])
    assert all(t["l1"] <= 0.04 + 1e-7 for t in turnover[1:])
    assert max(out["INSTRUMENT:1"]) <= 0.9 + 1e-7
    assert max(out["INSTRUMENT:2"]) <= 0.9 + 1e-7


@pytest.mark.parametrize("method,risk", [("gmv", None), ("mvo", 10.0)])
def test_future_prices_do_not_change_prior_targets(method, risk):
    latest, targets = panel()
    args = dict(method=method, max_weight=0.9, max_turnover=0.1, risk_aversion=risk)
    before, _ = _constrained_targets(targets, latest, 20, **args)
    for (_, i), row in latest.items():
        if int(i) > 40:
            row["close"] *= 3
    after, _ = _constrained_targets(targets, latest, 20, **args)
    np.testing.assert_allclose(
        before.head(21).drop("date"), after.head(21).drop("date"), atol=1e-10
    )


def test_zero_budget_holds_targets_and_missing_history_rejects():
    latest, targets = panel()
    out, _ = _constrained_targets(
        targets, latest, 20, method="mvo", max_weight=0.9, max_turnover=0.0, risk_aversion=10.0
    )
    assert max(out["INSTRUMENT:1"]) - min(out["INSTRUMENT:1"]) < 1e-7
    with pytest.raises(ValueError, match="complete trailing"):
        _constrained_targets(
            targets, latest, 100, method="gmv", max_weight=0.9, max_turnover=0.1, risk_aversion=None
        )


def test_solver_failure_rejects_entire_allocation(monkeypatch):
    import cvxpy as cp

    def fail(*args, **kwargs):
        raise cp.error.SolverError("unavailable")

    monkeypatch.setattr(cp.Problem, "solve", fail)
    with pytest.raises(ValueError, match="solver failed"):
        allocate(*panel())


def test_mvo_nonzero_mean_matches_independent_two_asset_solution():
    latest, targets = panel()
    for (instrument, i), row in latest.items():
        if instrument == "2":
            row["close"] *= 1.0001 ** int(i)
    prices = np.array([[latest[str(j), str(i)]["close"] for j in [1, 2]] for i in range(21)])
    returns = prices[1:] / prices[:-1] - 1
    cov = np.cov(returns, rowvar=False, ddof=1)
    mu = returns.mean(axis=0)
    gamma = 10.0
    oracle = (cov[1, 1] - cov[0, 1] + (mu[0] - mu[1]) / (2 * gamma)) / (
        cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    )
    out, _ = _constrained_targets(
        targets.head(1),
        latest,
        20,
        method="mvo",
        max_weight=0.9,
        max_turnover=0.1,
        risk_aversion=gamma,
    )
    assert 0.1 < oracle < 0.9
    assert out["INSTRUMENT:1"][0] == pytest.approx(oracle, abs=1e-5)
