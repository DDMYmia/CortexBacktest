from datetime import datetime, timedelta, timezone

import numpy as np
import polars as pl
import pytest

from cortexbacktest.reporting.risk_calibration import historical_var_calibration


def curve(n=100):
    returns = np.linspace(-0.03, 0.04, n)
    return pl.DataFrame(
        {
            "timestamp": [
                datetime(2025, 1, 1, tzinfo=timezone.utc) + timedelta(days=i) for i in range(n + 1)
            ],
            "equity": 100 * np.r_[1.0, np.cumprod(1 + returns)],
        }
    )


def calibrate(data):
    return historical_var_calibration(data, window=20, confidence=0.9, significance=0.05)


def test_rolling_var_linear_quantile_clock_and_coverage_match_independent_oracle():
    result = calibrate(curve())
    assert result["status"] == "available" and result["calibratedIntervals"] == 80
    assert result["totalReturnIntervals"] == 100 and result["warmupExcluded"] == 20
    row = result["observations"][0]
    original = np.linspace(-0.03, 0.04, 100)
    oracle = -(original[1] * 0.1 + original[2] * 0.9)
    assert row["var"] == pytest.approx(oracle, abs=1e-12)
    assert row["return"] == pytest.approx(original[20])
    assert row["forecast_at"] == row["period_start"]
    assert row["forecast_at"] < row["timestamp"]
    breaches = sum(r["return"] < -r["var"] for r in result["observations"])
    assert result["summary"]["violations"] == breaches
    assert not result["qualificationIssued"] and not result["executionAuthorized"]


def test_forecast_does_not_use_current_or_future_interval_return():
    original = curve()
    changed = original.with_columns(
        pl.when(pl.int_range(pl.len()) >= 50)
        .then(pl.col("equity") * 0.5)
        .otherwise(pl.col("equity"))
        .alias("equity")
    )
    a, b = calibrate(original), calibrate(changed)
    # Forecast ending at index50 may not see that interval's equity shock.
    assert [r["var"] for r in a["observations"][:30]] == [r["var"] for r in b["observations"][:30]]
    assert a["observations"][29]["return"] != b["observations"][29]["return"]


def test_insufficient_warmup_or_scored_sample_never_reports_calibration_pass():
    for n in [1, 10, 30, 49]:
        result = calibrate(curve(n))
        assert result["status"] == "unavailable" and result["summary"] is None
        assert result["observations"] == [] and result["calibratedIntervals"] == 0
        assert result["reason"] == "insufficient_post_warmup_observations"


def test_constant_cash_returns_preserve_unavailable_independence_test():
    result = calibrate(curve().with_columns(pl.lit(100.0).alias("equity")))
    assert result["summary"]["violations"] == 0
    assert result["summary"]["independence"]["pValue"] is None
    assert result["summary"]["independence"]["reason"] == "unobserved_transition_origin"
    assert all(row["var"] == 0.0 for row in result["observations"])
    assert result["warnings"]


@pytest.mark.parametrize("kind", ["nan", "zero", "negative", "duplicate", "naive"])
def test_bad_native_equity_is_rejected(kind):
    data = curve()
    if kind in ["nan", "zero", "negative"]:
        data = data.with_columns(
            pl.lit({"nan": float("nan"), "zero": 0.0, "negative": -1.0}[kind]).alias("equity")
        )
    elif kind == "duplicate":
        data = pl.concat([data, data.head(1)])
    else:
        data = data.with_columns(pl.col("timestamp").dt.replace_time_zone(None))
    with pytest.raises(ValueError):
        calibrate(data)


@pytest.mark.parametrize(
    "patch",
    [
        {"window": True},
        {"window": 19},
        {"confidence": True},
        {"confidence": 0.7},
        {"significance": float("nan")},
        {"significance": 0.3},
    ],
)
def test_invalid_risk_configuration_is_rejected(patch):
    with pytest.raises(ValueError):
        historical_var_calibration(
            curve(), **{"window": 20, "confidence": 0.95, "significance": 0.05, **patch}
        )
