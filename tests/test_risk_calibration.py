from datetime import datetime, timedelta, timezone
import json
import math

import numpy as np
import polars as pl
import pytest

from cortexbacktest.reporting import (christoffersen_independence, kupiec_pof,
                                     returns_from_equity, var_backtest)
from cortexbacktest.options.numerical import _solve_tridiagonal


def frames():
    dates = [datetime(2025, 1, 2, tzinfo=timezone.utc) + timedelta(days=i) for i in range(8)]
    starts = [t - timedelta(days=1) for t in dates]
    returns = pl.DataFrame({"timestamp": dates, "period_start": starts,
                           "return": [-.01, -.03, .01, -.04, -.05, .01, .01, -.02]})
    forecasts = pl.DataFrame({"timestamp": dates, "forecast_at": starts, "var": [.02] * 8})
    return returns, forecasts


def test_breaches_transitions_and_likelihood_formulas():
    returns, forecasts = frames()
    result = var_backtest(returns, forecasts, confidence=.8)
    assert result.observations["breach"].to_list() == [False, True, False, True, True, False, False, False]
    counts = result.summary["independence"]["transitions"]
    assert counts == {"n00": 2, "n01": 2, "n10": 2, "n11": 1}
    expected = 2 * (3 * math.log((3/8)/.2) + 5 * math.log((5/8)/.8))
    assert result.summary["unconditionalCoverage"]["statistic"] == pytest.approx(expected)
    total = expected + result.summary["independence"]["statistic"]
    assert result.summary["conditionalCoverage"]["pValue"] == pytest.approx(math.exp(-total/2))
    json.dumps(result.to_dict(), allow_nan=False)


def test_zero_and_all_exceptions_are_finite_not_independence_approval():
    for x in [0, 100]:
        assert math.isfinite(kupiec_pof(x, 100)["statistic"])
    for breaches in [[False] * 100, [True] * 100]:
        result = christoffersen_independence(breaches)
        assert result["pValue"] is None and result["reason"] == "unobserved_transition_origin"


@pytest.mark.parametrize("kind", ["missing", "extra", "duplicate", "late", "negative", "null", "nan", "naive", "zero_interval"])
def test_reject_invalid_scope_clocks_and_values(kind):
    returns, forecasts = frames()
    if kind == "missing":
        forecasts = forecasts.slice(1)
    elif kind == "extra":
        returns = returns.slice(1)
    elif kind == "duplicate":
        forecasts = pl.concat([forecasts, forecasts.head(1)])
    elif kind == "late":
        forecasts = forecasts.with_columns((pl.col("forecast_at") + timedelta(hours=1)))
    elif kind == "negative":
        forecasts = forecasts.with_columns(pl.lit(-.01).alias("var"))
    elif kind == "null":
        forecasts = forecasts.with_columns(pl.lit(None, dtype=pl.Float64).alias("var"))
    elif kind == "nan":
        returns = returns.with_columns(pl.lit(float("nan")).alias("return"))
    elif kind == "naive":
        forecasts = forecasts.with_columns(pl.col("forecast_at").dt.replace_time_zone(None))
    else:
        returns = returns.with_columns(pl.col("timestamp").alias("period_start"))
    with pytest.raises(ValueError):
        var_backtest(returns, forecasts)


def test_native_equity_returns_feed_calibration():
    observations, forecasts = frames()
    observations = observations.with_columns(pl.when(pl.col("return") == -.02)
        .then(-.019).otherwise(pl.col("return")).alias("return"))
    equity = pl.DataFrame({"timestamp": observations["timestamp"],
                          "equity": 100 * np.cumprod(1 + observations["return"].to_numpy())})
    returns = returns_from_equity(equity, initial_equity=100.).with_columns(observations["period_start"])
    assert var_backtest(returns, forecasts).summary["violations"] == 3


@pytest.mark.parametrize("confidence", [0, 1, float("nan"), True])
def test_confidence_is_strict(confidence):
    with pytest.raises(ValueError):
        kupiec_pof(1, 100, confidence=confidence)


@pytest.mark.parametrize("n", [1, 2, 10, 50])
def test_native_tridiagonal_matches_dense_oracle_with_pivoting(n):
    rng = np.random.default_rng(45 + n)
    for _ in range(20):
        bands = rng.normal(size=(3, n))
        if n > 1:
            bands[1, 0] = 0  # Force the adjacent-row pivot path.
        dense = np.diag(bands[1]) + np.diag(bands[0, 1:], 1) + np.diag(bands[2, :-1], -1)
        rhs = rng.normal(size=n)
        actual = _solve_tridiagonal(bands, rhs)
        np.testing.assert_allclose(actual, np.linalg.solve(dense, rhs), rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(dense @ actual, rhs, rtol=1e-10, atol=1e-10)


def test_singular_tridiagonal_is_rejected():
    with pytest.raises(np.linalg.LinAlgError):
        _solve_tridiagonal(np.zeros((3, 3)), np.ones(3))


def test_overlapping_or_gapped_observation_intervals_are_rejected():
    returns, forecasts = frames()
    returns = returns.with_columns((pl.col("period_start") - timedelta(hours=1)))
    forecasts = forecasts.with_columns((pl.col("forecast_at") - timedelta(hours=2)))
    with pytest.raises(ValueError, match="contiguous"):
        var_backtest(returns, forecasts)


def test_clock_precision_alignment_and_extreme_valid_confidence():
    returns, forecasts = frames()
    forecasts = forecasts.with_columns(pl.col("timestamp").dt.cast_time_unit("ns"))
    assert var_backtest(returns, forecasts).summary["violations"] == 3
    assert math.isfinite(kupiec_pof(1, 100, confidence=1e-300)["statistic"])


def test_nanosecond_gap_is_not_hidden_by_python_datetime_rounding():
    returns, forecasts = frames()
    returns = returns.with_columns(pl.col("timestamp", "period_start").dt.cast_time_unit("ns"))
    returns = returns.with_columns(pl.when(pl.int_range(pl.len()) == 1)
        .then(pl.col("period_start") + pl.duration(nanoseconds=1))
        .otherwise(pl.col("period_start")).alias("period_start"))
    with pytest.raises(ValueError, match="contiguous"):
        var_backtest(returns, forecasts)
