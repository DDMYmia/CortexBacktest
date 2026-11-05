from datetime import UTC, date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl
import pytest

from cortexbacktest.reporting.risk_model import estimate_style_risk

KNOWN = datetime(2025, 12, 25, tzinfo=UTC)


def inputs(count=10):
    # Independent synthetic numerical oracle on a 60-session grid, including 21h close-to-close.
    start, end = date(2025, 10, 1), date(2025, 12, 24)
    days = [start + timedelta(days=i) for i in range((end - start).days + 1)]
    days = [d for d in days if d.weekday() < 5 and d != date(2025, 11, 27)]
    dates = [
        datetime.combine(
            d,
            time(13) if d in {date(2025, 11, 28), date(2025, 12, 24)} else time(16),
            ZoneInfo("America/New_York"),
        ).astimezone(UTC)
        for d in days
    ]
    assert len(dates) == 60 and dates[-1] - dates[-2] == timedelta(hours=21)
    size = np.linspace(-1, 1, count)
    value = size**2 - np.mean(size**2)
    X = np.column_stack([np.ones(count), size, value])
    t = np.arange(60)
    f = np.column_stack(
        [np.sin(t / 5) * 0.002 + 0.001, np.cos(t / 3) * 0.001, np.sin(t / 7) * 0.003]
    )
    z = np.cos(np.arange(count))
    orthogonal = z - X @ np.linalg.lstsq(X, z, rcond=None)[0]
    residual = np.outer(np.cos(t / 4) * 0.0005, orthogonal)
    Y = f @ X.T + residual
    rows = [
        {"timestamp": stamp, "known_at": KNOWN, "symbol": str(i + 1), "return": float(Y[j, i])}
        for j, stamp in enumerate(dates)
        for i in range(count)
    ]
    styles = pl.DataFrame(
        [
            {
                "timestamp": KNOWN,
                "known_at": KNOWN,
                "symbol": str(i + 1),
                "size": size[i],
                "value": value[i],
            }
            for i in range(count)
        ]
    )
    return (
        pl.DataFrame(rows),
        styles,
        np.cov(f, rowvar=False, ddof=1),
        np.var(residual, axis=0, ddof=1),
    )


def fit(returns=None, styles=None):
    r, s, _, _ = inputs()
    r = r if returns is None else returns
    s = s if styles is None else styles
    return estimate_style_risk(
        r,
        s,
        fit_end=r["timestamp"].max(),
        model_known_at=KNOWN,
        currency="USD",
        input_sha256="b" * 64,
        model_id="style-oracle",
    )


def test_known_factor_and_specific_covariance_oracle_with_half_day():
    returns, styles, F, D = inputs()
    result = fit(returns, styles)
    np.testing.assert_allclose(result["model"]["covariance"], F, atol=1e-18, rtol=1e-10)
    for asset in result["model"]["assets"]:
        assert asset["specific_variance"] == pytest.approx(D[asset["instrument_id"] - 1], abs=1e-18)
        assert asset["loadings"][0] == 1.0
    assert result["model"]["factors"] == ["market", "size", "value"]
    assert result["fit"]["designRank"] == 3 and result["fit"]["calendar"] == "us_equity"
    assert result["fit"]["specificFactorCovariance"] == "assumed_zero"
    assert not result["fit"]["qualificationIssued"]
    assert fit(returns.reverse(), styles.reverse()) == result


@pytest.mark.parametrize(
    "case",
    [
        "missing_return",
        "missing_asset",
        "rank",
        "future_exposure",
        "future_receipt",
        "intraday",
        "reserved",
    ],
)
def test_incomplete_rank_deficient_and_future_inputs_never_produce_model(case):
    returns, styles, _, _ = inputs()
    if case == "missing_return":
        returns = returns.slice(1)
    elif case == "missing_asset":
        returns = returns.filter(pl.col("symbol") != "10")
        styles = styles.filter(pl.col("symbol") != "10")
    elif case == "rank":
        styles = styles.with_columns(pl.col("size").alias("value"))
    elif case == "future_exposure":
        styles = styles.with_columns(pl.lit(KNOWN + timedelta(seconds=1)).alias("timestamp"))
    elif case == "future_receipt":
        returns = returns.with_columns(pl.lit(KNOWN + timedelta(seconds=1)).alias("known_at"))
    elif case == "intraday":
        returns = returns.with_columns(
            (pl.col("timestamp") - timedelta(minutes=30)).alias("timestamp")
        )
    else:
        styles = styles.rename({"value": "weight"})
    with pytest.raises(ValueError):
        fit(returns, styles)


def test_omitted_terms_match_independent_joint_covariance_oracle():
    # Known synthetic factor/residual generator, without fitting another model.
    size = np.linspace(-1, 1, 10)
    X = np.column_stack([np.ones(10), size, size**2 - np.mean(size**2)])
    t = np.arange(60)
    factors = np.column_stack(
        [np.sin(t / 5) * 0.002 + 0.001, np.cos(t / 3) * 0.001, np.sin(t / 7) * 0.003]
    )
    z = np.cos(np.arange(10))
    residual_direction = z - X @ np.linalg.lstsq(X, z, rcond=None)[0]
    residual = np.outer(np.cos(t / 4) * 0.0005, residual_direction)
    joint = np.cov(np.column_stack([factors, residual]), rowvar=False, ddof=1)
    off = joint[3:, 3:] - np.diag(np.diag(joint[3:, 3:]))
    cross = X @ joint[:3, 3:] + joint[3:, :3] @ X.T
    empirical = np.cov(factors @ X.T + residual, rowvar=False, ddof=1)
    scale = np.linalg.norm(empirical, "fro")
    diag = fit()["fit"]["covarianceDiagnostics"]
    assert diag["residualOffDiagonalRelativeNorm"] == pytest.approx(
        np.linalg.norm(off, "fro") / scale
    )
    assert diag["factorSpecificCrossRelativeNorm"] == pytest.approx(
        np.linalg.norm(cross, "fro") / scale
    )
    assert diag["factorSpecificCrossRelativeNorm"] > 0
    assert (
        diag["residualOffDiagonalSignedGapProjection"]
        + diag["factorSpecificCrossSignedGapProjection"]
    ) == pytest.approx(1)
    assert diag["reconstructionMaxAbsError"] < 1e-18
    assert diag["impliedMinEigenvalue"] >= 0


def test_zero_empirical_covariance_keeps_undefined_ratios_null():
    returns, styles, _, _ = inputs()
    result = fit(returns.with_columns(pl.lit(0.0).alias("return")), styles)
    assert result["fit"]["relativeEmpiricalCovarianceGap"] is None
    diag = result["fit"]["covarianceDiagnostics"]
    assert diag["residualOffDiagonalRelativeNorm"] is None
    assert diag["factorSpecificCrossRelativeNorm"] is None
    assert diag["residualOffDiagonalSignedGapProjection"] is None
    assert diag["factorSpecificCrossSignedGapProjection"] is None
    assert diag["reconstructionMaxAbsError"] == 0


@pytest.mark.parametrize("count", [11, 50, 100])
def test_bounded_style_universe_matches_independent_covariance_oracle(count):
    returns, styles, covariance, specific = inputs(count)
    result = fit(returns, styles)
    assert len(result["model"]["assets"]) == count
    np.testing.assert_allclose(result["model"]["covariance"], covariance, atol=1e-18, rtol=1e-10)
    for asset in result["model"]["assets"]:
        assert asset["specific_variance"] == pytest.approx(specific[asset["instrument_id"] - 1], abs=1e-18)


def test_style_universe_rejects_above_bound():
    returns, styles, _, _ = inputs(101)
    with pytest.raises(ValueError, match="bounded stable"):
        fit(returns, styles)
