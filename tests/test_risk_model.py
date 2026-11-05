from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from cortexbacktest.reporting.risk_model import estimate_market_risk

START = datetime(2025, 1, 1, tzinfo=UTC)
END = START + timedelta(days=59)
KNOWN = END + timedelta(hours=2)


def data():
    return pl.DataFrame(
        [
            {
                "timestamp": START + timedelta(days=i),
                "symbol": symbol,
                "known_at": KNOWN,
                "return": (i - 30) / 10000 * beta + alpha,
            }
            for symbol, beta, alpha in [("1", 1, 0), ("2", 2, 0.003), ("3", 0, 0.001)]
            for i in range(60)
        ]
    )


def fit(frame=None, **kwargs):
    return estimate_market_risk(
        data() if frame is None else frame,
        benchmark="1",
        fit_end=END,
        model_known_at=KNOWN,
        currency="USD",
        input_sha256="a" * 64,
        model_id="market-test",
        **kwargs,
    )


def test_ols_independent_oracle_covariance_and_zero_specific_risk():
    result = fit()
    expected = np.var(np.array([(i - 30) / 10000 for i in range(60)]), ddof=1)
    assert result["model"]["covariance"][0][0] == pytest.approx(expected)
    assets = {a["instrument_id"]: a for a in result["model"]["assets"]}
    assert assets[1]["loadings"] == pytest.approx([1])
    assert assets[2]["loadings"] == pytest.approx([2])
    assert assets[3]["loadings"] == pytest.approx([0], abs=1e-12)
    assert all(a["specific_variance"] < 1e-25 for a in assets.values())
    assert result["fit"]["intercepts"]["2"] == pytest.approx(0.003)
    assert result["fit"]["observations"] == 60
    assert not result["fit"]["qualificationIssued"]
    assert fit(data().reverse()) == result


@pytest.mark.parametrize(
    "case", ["missing", "duplicate", "future_receipt", "future_close", "nonfinite", "intraday"]
)
def test_missing_future_duplicate_and_wrong_frequency_fail(case):
    frame = data()
    if case == "missing":
        frame = frame.slice(1)
    elif case == "duplicate":
        frame = pl.concat([frame, frame.head(1)])
    elif case == "future_receipt":
        frame = frame.with_columns(pl.lit(KNOWN + timedelta(seconds=1)).alias("known_at"))
    elif case == "future_close":
        frame = frame.with_columns((pl.col("timestamp") + timedelta(days=1)).alias("timestamp"))
    elif case == "nonfinite":
        frame = frame.with_columns(pl.lit(float("nan")).alias("return"))
    else:
        frame = frame.with_columns(
            (pl.lit(START) + (pl.col("timestamp") - START) / 2).alias("timestamp")
        )
    with pytest.raises(ValueError):
        fit(frame)


def test_constant_market_and_insufficient_samples_refuse_estimation():
    with pytest.raises(ValueError, match="positive finite"):
        fit(data().with_columns(pl.lit(0.001).alias("return")))
    with pytest.raises(ValueError, match="insufficient"):
        fit(data().filter(pl.col("timestamp") != START))
    with pytest.raises(ValueError, match="contract"):
        fit(minimum_observations=3)
