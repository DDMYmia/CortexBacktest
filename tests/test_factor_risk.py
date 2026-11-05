from datetime import datetime, timedelta, timezone
import json
import math

import numpy as np
import polars as pl
import pytest

from cortexbacktest.reporting import (
    factor_risk_decomposition,
    portfolio_style_exposure,
    standardise_style_exposures,
)

T = datetime(2025, 1, 1, tzinfo=timezone.utc)
DEFS = {"small": {"log_cap": -1}, "value": {"book_price": 1, "earnings_price": 1}}


def raw():
    return pl.DataFrame(
        {
            "timestamp": [T] * 12,
            "known_at": [T - timedelta(days=1)] * 12,
            "symbol": [f"A{i:02d}" for i in range(12)],
            "market_cap": [10.0 ** (i / 3) for i in range(12)],
            "log_cap": [i / 3 * math.log(10) for i in range(12)],
            "book_price": [0.1 + i * 0.01 for i in range(12)],
            "earnings_price": [0.02 + (i % 5) * 0.005 for i in range(12)],
        }
    )


def model():
    weights = pl.DataFrame(
        {"timestamp": [T, T], "known_at": [T, T], "symbol": ["A", "B"], "weight": [0.6, 0.4]}
    )
    exposures = pl.DataFrame(
        {
            "timestamp": [T, T],
            "known_at": [T, T],
            "symbol": ["A", "B"],
            "market": [1.0, 1.0],
            "value": [2.0, -1.0],
        }
    )
    cov = pl.DataFrame(
        {"factor": ["market", "value"], "market": [0.04, 0.01], "value": [0.01, 0.09]}
    )
    specific = pl.DataFrame(
        {"timestamp": [T, T], "known_at": [T, T], "symbol": ["A", "B"], "variance": [0.01, 0.02]}
    )
    return weights, exposures, cov, specific


def risk(*frames, **kwargs):
    return factor_risk_decomposition(
        *frames,
        as_of=T,
        model_known_at=T,
        fit_end=T - timedelta(days=1),
        model_id="fixed-academic-test-v1",
        currency="USD",
        horizon="daily",
        **kwargs,
    )


def test_cap_centered_styles_match_independent_numpy_definition_without_imputation():
    data = raw()
    result = standardise_style_exposures(data, definitions=DEFS)
    caps = data["market_cap"].to_numpy()
    caps = caps / caps.sum()
    for factor, components in DEFS.items():
        expected = np.zeros(12)
        for field, sign in components.items():
            values = data[field].to_numpy()
            low, high = np.quantile(values, [0.025, 0.975])
            values = np.clip(values, low, high)
            expected += (
                sign * (values - np.sum(caps * values)) / np.std(values, ddof=1) / len(components)
            )
        np.testing.assert_allclose(result[factor].to_numpy(), expected, atol=1e-14)
        assert abs(caps @ result[factor].to_numpy()) < 1e-14
    assert result.height == 12


@pytest.mark.parametrize(
    "kind", ["missing", "null", "nan", "small", "constant", "cap", "late", "duplicate", "collision"]
)
def test_standardisation_rejects_missing_universe_or_characteristics(kind):
    data = raw()
    defs = DEFS
    if kind == "missing":
        data = data.drop("book_price")
    elif kind == "null":
        data = data.with_columns(pl.lit(None, dtype=pl.Float64).alias("book_price"))
    elif kind == "nan":
        data = data.with_columns(pl.lit(float("nan")).alias("book_price"))
    elif kind == "small":
        data = data.head(6)
    elif kind == "constant":
        data = data.with_columns(pl.lit(0.2).alias("book_price"))
    elif kind == "cap":
        data = data.with_columns(pl.lit(0.0).alias("market_cap"))
    elif kind == "late":
        data = data.with_columns(pl.col("known_at") + timedelta(days=3))
    elif kind == "duplicate":
        data = pl.concat([data, data.head(1)])
    elif kind == "collision":
        defs = {"weight": {"book_price": 1}}
    with pytest.raises(ValueError):
        standardise_style_exposures(data, definitions=defs)


def test_style_history_active_drift_and_first_observation_undefined_sample_std():
    exposures = standardise_style_exposures(raw(), definitions=DEFS)
    weights = exposures.select("timestamp", "symbol", "known_at").with_columns(
        pl.lit(1 / 12).alias("weight"), pl.lit(1 / 12).alias("benchmark_weight")
    )
    first = portfolio_style_exposure(weights, exposures, model_id="frozen-definitions")
    assert first.drift["std_active_exposure"].null_count() == 2
    second = exposures.with_columns(pl.col("timestamp") + timedelta(days=1))
    next_weights = weights.with_columns(
        pl.col("timestamp") + timedelta(days=1), pl.Series("weight", [1.0] + [0.0] * 11)
    )
    report = portfolio_style_exposure(
        pl.concat([weights, next_weights], how="vertical_relaxed"),
        pl.concat([exposures, second], how="vertical_relaxed"),
        model_id="frozen-definitions",
    )
    for factor in DEFS:
        row = report.drift.filter(pl.col("factor") == factor).row(0, named=True)
        expected = float(exposures[factor][0] - exposures[factor].mean())
        assert row["total_change"] == pytest.approx(expected)
        assert row["std_active_exposure"] == pytest.approx(abs(expected) / math.sqrt(2))
        assert row["max_abs_step"] == pytest.approx(abs(expected))
    json.dumps(report.to_dict(), allow_nan=False)
    with pytest.raises(ValueError, match="exactly"):
        portfolio_style_exposure(weights, exposures.head(11), model_id="fixed")


def test_factor_risk_matches_dense_covariance_euler_and_finite_difference():
    w, x, F, d = model()
    result = risk(w, x, F, d)
    W = w["weight"].to_numpy()
    X = x.select("market", "value").to_numpy()
    cov = F.select("market", "value").to_numpy()
    D = d["variance"].to_numpy()
    dense = X @ cov @ X.T + np.diag(D)
    vol = math.sqrt(W @ dense @ W)
    assert result["totalVariance"] == pytest.approx(W @ dense @ W)
    assert result["factorVariance"] == pytest.approx(0.1136)
    assert result["specificVariance"] == pytest.approx(0.0068)
    contributions = np.array([r["volatilityContribution"] for r in result["assets"]])
    np.testing.assert_allclose(contributions, W * (dense @ W) / vol)
    for i in range(2):
        delta = np.zeros(2)
        delta[i] = 1e-6
        gradient = (
            math.sqrt((W + delta) @ dense @ (W + delta))
            - math.sqrt((W - delta) @ dense @ (W - delta))
        ) / 2e-6
        assert contributions[i] == pytest.approx(W[i] * gradient, rel=1e-8)
    assert sum(r["volatilityContribution"] for r in result["factors"]) + sum(
        r["specificVolatilityContribution"] for r in result["assets"]
    ) == pytest.approx(vol)
    assert sum(r["riskShare"] for r in result["assets"]) == pytest.approx(1.0)
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize(
    "kind",
    [
        "missing_asset",
        "missing_specific",
        "negative_specific",
        "negative_cov",
        "asymmetric",
        "missing_factor",
        "future",
        "fit",
        "null",
        "duplicate",
        "wrong_date",
        "collision",
    ],
)
def test_factor_risk_rejects_incomplete_or_future_models(kind):
    w, x, F, d = model()
    if kind == "missing_asset":
        x = x.head(1)
    elif kind == "missing_specific":
        d = d.head(1)
    elif kind == "negative_specific":
        d = d.with_columns(pl.lit(-0.1).alias("variance"))
    elif kind == "negative_cov":
        F = F.with_columns(pl.Series("market", [-0.04, 0.01]))
    elif kind == "asymmetric":
        F = F.with_columns(pl.Series("market", [0.04, 0.02]))
    elif kind == "missing_factor":
        F = F.drop("value")
    elif kind == "future":
        x = x.with_columns(pl.col("known_at") + timedelta(microseconds=1))
    elif kind == "fit":
        with pytest.raises(ValueError, match="available"):
            factor_risk_decomposition(
                w,
                x,
                F,
                d,
                as_of=T,
                model_known_at=T,
                fit_end=T + timedelta(days=1),
                model_id="test",
                currency="USD",
                horizon="daily",
            )
        return
    elif kind == "null":
        x = x.with_columns(pl.lit(None, dtype=pl.Float64).alias("value"))
    elif kind == "duplicate":
        w = pl.concat([w, w.head(1)])
    elif kind == "wrong_date":
        d = d.with_columns(
            pl.col("timestamp") - timedelta(days=1), pl.col("known_at") - timedelta(days=1)
        )
    elif kind == "collision":
        x = x.rename({"value": "weight"})
    with pytest.raises(ValueError):
        risk(w, x, F, d)


def test_zero_risk_shares_undefined_and_hedge_contributions_can_be_negative():
    w, x, F, d = model()
    F = F.with_columns(pl.lit(0.0).alias("market"), pl.lit(0.0).alias("value"))
    d = d.with_columns(pl.lit(0.0).alias("variance"))
    result = risk(w, x, F, d)
    assert result["riskShareStatus"] == "undefined_zero_risk"
    assert all(r["riskShare"] is None for r in result["assets"])
    w, x, F, d = model()
    w = w.with_columns(pl.Series("weight", [1.0, -0.2]))
    x = x.with_columns(pl.lit(1.0).alias("value"))
    result = risk(w, x, F, d)
    assert result["assets"][1]["volatilityContribution"] < 0
    assert sum(r["volatilityContribution"] for r in result["assets"]) == pytest.approx(
        result["volatility"]
    )


def test_risk_permutation_invariant_and_only_roundoff_psd_repair_reported():
    w, x, F, d = model()
    assert risk(
        w.reverse(), x.reverse(), F.reverse().select("value", "factor", "market"), d.reverse()
    ) == risk(w, x, F, d)
    F = F.with_columns(pl.Series("market", [1.0, 1.0]), pl.Series("value", [1.0, 1.0 - 1e-14]))
    result = risk(w, x, F, d)
    assert 0 < result["covarianceRoundoffAdjustment"] < 1e-12


def test_standardised_exposure_availability_includes_latest_universe_member():
    data = raw().with_columns(
        pl.when(pl.col("symbol") == "A11")
        .then(pl.lit(T))
        .otherwise(pl.col("known_at"))
        .alias("known_at")
    )
    result = standardise_style_exposures(data, definitions=DEFS)
    assert result["known_at"].n_unique() == 1
    assert result["known_at"][0] == T
