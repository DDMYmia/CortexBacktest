from datetime import datetime, timedelta, timezone
import json
from dataclasses import replace

import numpy as np
import polars as pl
import pytest

from cortexbacktest.research.recipes import (
    ResidualMomentumConfig,
    residual_momentum,
    group_budget_weights,
)
from cortexbacktest.research.pairs import PairConfig, PairModel, PairRecipe, fit_pairs

START = datetime(2020, 1, 1, tzinfo=timezone.utc)


def dates(n):
    return [START + timedelta(days=i) for i in range(n)]


def panel(n=140):
    rng = np.random.default_rng(42)
    f = rng.normal(0, 0.01, n)
    noise = rng.normal(0, 0.001, n)
    d = dates(n)
    return (
        pl.DataFrame({"date": d, "A": 0.002 + 2 * f + noise}),
        pl.DataFrame({"date": d, "known_at": d, "MKT": f}),
    )


def test_residual_out_of_fit_oracle_and_long_warmup():
    y, x = panel()
    cfg = ResidualMomentumConfig(110, 5, 2)
    result = residual_momentum(y, x, config=cfg, return_basis="excess")
    t = 130
    end = t + 1 - 2
    begin = end - 5
    design = np.column_stack([np.ones(110), x["MKT"].to_numpy()[begin - 110 : begin]])
    beta = np.linalg.lstsq(design, y["A"].to_numpy()[begin - 110 : begin], rcond=None)[0]
    resid = y["A"].to_numpy()[begin:end] - beta[0] - beta[1] * x["MKT"].to_numpy()[begin:end]
    assert result.values["A"][t] == pytest.approx(resid.sum() / resid.std(ddof=1))
    assert result.values["A"][:116].null_count() == 116
    assert result.diagnostics["fit_end"][t] < result.diagnostics["score_start"][t]
    json.dumps(result.to_dict(), allow_nan=False)


def test_residual_future_invariance_raw_rf_and_unavailable():
    y, x = panel(30)
    cfg = ResidualMomentumConfig(10, 5, 1)
    original = residual_momentum(y, x, config=cfg, return_basis="excess")
    changed = y.with_columns(
        pl.when(pl.int_range(pl.len()) > 20).then(99).otherwise(pl.col("A")).alias("A")
    )
    assert (
        residual_momentum(changed, x, config=cfg, return_basis="excess")
        .values.head(21)
        .equals(original.values.head(21))
    )
    late = x.with_columns(
        pl.when(pl.int_range(pl.len()) == 15)
        .then(pl.col("known_at") + timedelta(days=100))
        .otherwise(pl.col("known_at"))
        .alias("known_at")
    )
    assert (
        "factor_not_available_by_decision"
        in residual_momentum(y, late, config=cfg, return_basis="excess").diagnostics["reason"]
    )
    raw = y.with_columns(pl.col("A") + 0.0001)
    assert residual_momentum(
        raw, x.with_columns(pl.lit(0.0001).alias("RF")), config=cfg, return_basis="raw"
    ).values["A"].to_numpy() == pytest.approx(original.values["A"].to_numpy(), nan_ok=True)
    missing = y.with_columns(
        pl.when(pl.int_range(pl.len()) == 15).then(None).otherwise(pl.col("A")).alias("A")
    )
    assert (
        "missing_or_nonfinite_window"
        in residual_momentum(missing, x, config=cfg, return_basis="excess").diagnostics["reason"]
    )
    constant = x.with_columns(pl.lit(1.0).alias("MKT"))
    assert (
        "rank_deficient_or_ill_conditioned_factors"
        in residual_momentum(y, constant, config=cfg, return_basis="excess").diagnostics["reason"]
    )
    exact = y.with_columns((0.002 + 2 * x["MKT"]).alias("A"))
    assert (
        "undefined_residual_dispersion"
        in residual_momentum(exact, x, config=cfg, return_basis="excess").diagnostics["reason"]
    )
    with pytest.raises(ValueError, match="risk-free"):
        residual_momentum(y, x, config=cfg, return_basis="raw")


def membership(symbols, groups):
    return pl.DataFrame(
        {
            "date": [START] * len(symbols),
            "known_at": [START] * len(symbols),
            "symbol": symbols,
            "group": groups,
        }
    )


def test_group_fixed_budgets_and_dated_membership():
    scores = pl.DataFrame({"date": [START], "A": [1.0], "B": [2.0], "C": [-1.0], "D": [None]})
    m = membership(["A", "B", "C", "D"], ["X", "X", "Y", "Y"])
    budgets = {"X": 0.6, "Y": 0.4}
    out = group_budget_weights(scores, m, budgets)
    assert out.values.row(0, named=True) == {"date": START, "A": 0.0, "B": 0.6, "C": 0.0, "D": 0.0}
    ls = group_budget_weights(scores, m, budgets, long_short=True)
    assert ls.values["A"][0] == -0.3 and ls.values["B"][0] == 0.3
    assert ls.values["C"][0] == 0
    with pytest.raises(ValueError, match="no PIT"):
        group_budget_weights(scores, m.filter(pl.col("symbol") != "A"), budgets)
    with pytest.raises(ValueError, match="known by decision"):
        group_budget_weights(
            scores, m.with_columns(pl.col("known_at") + timedelta(days=1)), budgets
        )
    with pytest.raises(ValueError, match="gross"):
        group_budget_weights(scores, m, {"X": 0.8, "Y": 0.8})


def formation():
    n = 120
    rng = np.random.default_rng(31)
    x = 4 + np.cumsum(rng.normal(0.001, 0.01, n))
    noise = rng.normal(0, 0.005, n)
    return pl.DataFrame(
        {
            "date": dates(n),
            "A": np.exp(0.2 + 1.2 * x + noise),
            "B": np.exp(x),
            "C": np.exp(2 + np.cumsum(rng.normal(0, 0.08, n))),
        }
    )


def test_pair_formation_distance_ols_and_frozen_binding():
    data = formation()
    cfg = PairConfig(max_pairs=1)
    fitted = fit_pairs(data, config=cfg, known_at=data["date"][-1], currency="USD")
    assert len(fitted.models) == 1
    model = fitted.models[0]
    assert (model.left, model.right) == ("A", "B")
    assert model.beta == pytest.approx(
        np.linalg.lstsq(
            np.column_stack([np.ones(120), np.log(data["B"])]), np.log(data["A"]), rcond=None
        )[0][1]
    )
    assert model.distance == pytest.approx(
        np.sum((data["A"].to_numpy() / data["A"][0] - data["B"].to_numpy() / data["B"][0]) ** 2)
    )
    with pytest.raises(ValueError, match="exact history"):
        fit_pairs(data.head(119), config=cfg, known_at=data["date"][-1], currency="USD")
    zero = data.with_columns(pl.lit(1.0).alias("A"), pl.lit(1.0).alias("B"), pl.lit(1.0).alias("C"))
    assert not fit_pairs(zero, config=cfg, known_at=data["date"][-1], currency="USD").models


def model():
    return PairModel(
        "A", "B", 0.0, 2.0, 0.0, 1.0, 0.0, START - timedelta(days=119), START, START, "0" * 64
    )


def quote(z):
    return {"A": float(np.exp(z)), "B": 1.0}


def test_pair_gate_confirmation_age_stop_and_mean_cross():
    m = model()
    cfg = PairConfig(max_pairs=1, confirmation_periods=2, cooldown_periods=1, max_holding_periods=2)
    r = PairRecipe((m,), config=cfg)

    def step(i, z):
        return r.update(START + timedelta(days=i), quote(z))

    assert step(1, 2.5) == {"A": 0.0, "B": 0.0}
    assert step(2, 2.5) == pytest.approx({"A": -1 / 3, "B": 2 / 3})
    assert step(3, 1.0) == pytest.approx({"A": -1 / 3, "B": 2 / 3})
    assert step(4, 2.5) == {"A": 0.0, "B": 0.0}  # age expiry
    assert step(5, 2.5) == {"A": 0.0, "B": 0.0}  # cooldown
    step(6, 2.5)
    assert step(7, 2.5)["A"] < 0
    assert step(8, -0.1) == {"A": 0.0, "B": 0.0}  # crossed mean
    step(9, 2.5)
    step(10, 2.5)
    step(11, 2.5)
    assert step(12, 4.0) == {"A": 0.0, "B": 0.0}  # stop boundary


def test_pair_all_checkpoint_cuts_and_tamper_rejection():
    m = model()
    cfg = PairConfig(max_pairs=2, cooldown_periods=1)
    zs = [2.5, 1.0, 0.4, 2.5, 2.5, None, 2.5, 2.5, -2.5, -1.0, -0.1]

    def replay(r, begin, end):
        return [
            r.update(
                START + timedelta(days=i + 1),
                quote(zs[i]) if zs[i] is not None else {"A": None, "B": 1.0},
            )
            for i in range(begin, end)
        ]

    whole = PairRecipe((m,), config=cfg)
    expected = replay(whole, 0, len(zs))
    assert sum(abs(v) for v in expected[0].values()) == pytest.approx(0.5)  # spare slot cash
    for cut in range(len(zs) + 1):
        left = PairRecipe((m,), config=cfg)
        before = replay(left, 0, cut)
        doc = json.loads(json.dumps(left.to_document()))
        restored = PairRecipe.from_document(doc, models=(m,), config=cfg)
        assert before + replay(restored, cut, len(zs)) == expected
        assert restored.to_document() == whole.to_document()
        with pytest.raises(ValueError, match="binding"):
            PairRecipe.from_document(doc, models=(replace(m, beta=1.0),), config=cfg)
    with pytest.raises(ValueError, match="strictly follow"):
        PairRecipe((m,), config=cfg).update(START, quote(2.5))


def test_pair_shared_asset_netting_and_invalid_event_atomicity():
    a = model()
    b = replace(a, left="B", right="C", beta=1.0)
    cfg = PairConfig(max_pairs=2)
    r = PairRecipe((a, b), config=cfg)
    prices = {"A": float(np.exp(7.5)), "B": float(np.exp(2.5)), "C": 1.0}
    out = r.update(START + timedelta(days=1), prices)
    assert out == pytest.approx({"A": -1 / 6, "B": 1 / 3 - 1 / 4, "C": 1 / 4})
    saved = r.to_document()
    with pytest.raises(ValueError):
        r.update(START + timedelta(days=1), prices)
    assert r.to_document() == saved
    with pytest.raises(ValueError):
        r.update(START + timedelta(days=2), {"A": -1.0, "B": 1.0, "C": 1.0})
    assert r.to_document() == saved


def test_optional_cointegration_unavailable_is_explicit(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "statsmodels.tsa.stattools", None)
    data = formation()
    cfg = PairConfig(max_pairs=1)
    result = fit_pairs(
        data, config=cfg, known_at=data["date"][-1], currency="USD", cointegration=True
    )
    assert set(result.candidates.filter(pl.col("reason").is_null())["cointegration_status"]) == {
        "dependency_unavailable"
    }
    assert result.candidates["cointegration_pvalue"].null_count() == result.candidates.height


def test_optional_cointegration_and_half_life_direct_oracle():
    sm = pytest.importorskip("statsmodels.tsa.stattools")
    data = formation()
    cfg = PairConfig(max_pairs=1)
    result = fit_pairs(
        data, config=cfg, known_at=data["date"][-1], currency="USD", cointegration=True
    )
    baseline = fit_pairs(data, config=cfg, known_at=data["date"][-1], currency="USD")
    assert result.models == baseline.models
    for row in result.candidates.filter(pl.col("reason").is_null()).iter_rows(named=True):
        y = np.log(data[row["left"]].to_numpy())
        x = np.log(data[row["right"]].to_numpy())
        stat, pvalue, _ = sm.coint(y, x, trend="c", maxlag=1, autolag=None)
        assert row["cointegration_status"] == "available_diagnostic_only"
        assert row["cointegration_statistic"] == pytest.approx(stat)
        assert row["cointegration_pvalue"] == pytest.approx(pvalue)
        coef = np.linalg.lstsq(np.column_stack([np.ones(120), x]), y, rcond=None)[0]
        spread = y - coef[0] - coef[1] * x
        phi = np.linalg.lstsq(np.column_stack([np.ones(119), spread[:-1]]), spread[1:], rcond=None)[
            0
        ][1]
        assert row["ar1"] == pytest.approx(phi)
        if 0 < phi < 1:
            assert row["half_life_periods"] == pytest.approx(np.log(0.5) / np.log(phi))
        else:
            assert row["half_life_periods"] is None


@pytest.mark.parametrize("z", [0.5, 0.0, -1.0, 4.0, -4.0])
def test_pair_exit_boundaries(z):
    r = PairRecipe((model(),), config=PairConfig(max_pairs=1))
    r.update(START + timedelta(days=1), quote(2.5))
    assert r.update(START + timedelta(days=2), quote(z)) == {"A": 0.0, "B": 0.0}


def test_future_factor_does_not_change_past_scores():
    y, x = panel(40)
    cfg = ResidualMomentumConfig(10, 5, 1)
    a = residual_momentum(y, x, config=cfg, return_basis="excess")
    x = x.with_columns(
        pl.when(pl.int_range(pl.len()) > 25).then(100.0).otherwise(pl.col("MKT")).alias("MKT")
    )
    b = residual_momentum(y, x, config=cfg, return_basis="excess")
    assert a.values.head(26).equals(b.values.head(26))


def test_residual_explicit_asset_receipt_clock():
    y, x = panel(40)
    cfg = ResidualMomentumConfig(10, 5, 1)
    late_y = y.with_columns(pl.col("date").alias("known_at")).with_columns(
        pl.when(pl.int_range(pl.len()) == 20)
        .then(pl.col("known_at") + timedelta(days=50))
        .otherwise(pl.col("known_at"))
        .alias("known_at")
    )
    result = residual_momentum(late_y, x, config=cfg, return_basis="excess")
    assert result.summary["assetAvailability"] == "explicit_known_at"
    assert result.values["A"][21] is None
    assert result.diagnostics["reason"][21] == "asset_not_available_by_decision"
    assert result.values.head(20).equals(
        residual_momentum(y, x, config=cfg, return_basis="excess").values.head(20)
    )
    with pytest.raises(ValueError, match="asset return availability"):
        residual_momentum(
            y.with_columns((pl.col("date") - timedelta(seconds=1)).alias("known_at")),
            x,
            config=cfg,
            return_basis="excess",
        )


def test_pair_formation_price_receipts_are_bound():
    data = formation().with_columns(pl.col("date").alias("known_at"))
    cfg = PairConfig(max_pairs=1)
    cutoff = data["date"][-1] + timedelta(days=1)
    a = fit_pairs(data, config=cfg, known_at=cutoff, currency="USD")
    assert a.summary["priceAvailability"] == "explicit_known_at"
    later = data.with_columns((pl.col("known_at") + timedelta(hours=1)).alias("known_at"))
    b = fit_pairs(later, config=cfg, known_at=cutoff, currency="USD")
    assert a.models[0].beta == b.models[0].beta
    assert a.models[0].data_sha256 != b.models[0].data_sha256
    with pytest.raises(ValueError, match="must be received"):
        fit_pairs(
            data.with_columns((pl.col("known_at") + timedelta(days=200)).alias("known_at")),
            config=cfg,
            known_at=cutoff,
            currency="USD",
        )
