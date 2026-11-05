"Daily single-market OLS covariance model, with explicit received-input clocks."

from __future__ import annotations

from datetime import datetime, time
import math
import re
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl


def _model_identity(model_id, currency, input_sha256):
    if (
        not re.fullmatch(r"[A-Za-z0-9._:-]{1,128}", model_id)
        or not re.fullmatch(r"[A-Z0-9]{2,8}", currency)
        or not re.fullmatch(r"[0-9a-f]{64}", input_sha256)
    ):
        raise ValueError("explicit bounded risk-model identity required")


def _return_panel(returns, *, fit_end, model_known_at, minimum_observations, calendar="crypto_utc", maximum_assets=10):
    if (
        set(returns.columns) != {"timestamp", "symbol", "known_at", "return"}
        or not returns.height
        or type(minimum_observations) is not int
        or minimum_observations < 60
    ):
        raise ValueError("explicit bounded risk-model contract required")
    for stamp in [fit_end, model_known_at]:
        if not isinstance(stamp, datetime) or stamp.tzinfo is None or stamp.utcoffset() is None:
            raise ValueError("aware model clocks required")
    if fit_end > model_known_at:
        raise ValueError("model fit cannot follow model availability")
    for field in ["timestamp", "known_at"]:
        dtype = returns.schema[field]
        if not isinstance(dtype, pl.Datetime) or not dtype.time_zone or returns[field].null_count():
            raise ValueError("complete aware return clocks required")
    if (
        returns.schema["symbol"] != pl.String
        or returns["symbol"].null_count()
        or not returns.schema["return"].is_numeric()
        or returns["return"].null_count()
        or not returns["return"].is_finite().all()
    ):
        raise ValueError("complete named finite returns required")
    data = returns.sort("timestamp", "symbol")
    symbols = sorted(data["symbol"].unique().to_list())
    if not 2 <= len(symbols) <= maximum_assets or any(
        not re.fullmatch(r"[1-9][0-9]*", symbol) for symbol in symbols
    ):
        raise ValueError("bounded stable instrument IDs and explicit market benchmark required")
    if data.select("timestamp", "symbol").unique().height != data.height:
        raise ValueError("duplicate market-model return")
    if (
        data.filter(pl.col("timestamp") > fit_end).height
        or data.filter(pl.col("known_at") > model_known_at).height
        or data.filter(pl.col("known_at") < pl.col("timestamp")).height
        or data.filter(pl.col("return") <= -1).height
    ):
        raise ValueError("returns must be received by model availability and close by fit end")
    stamps = data["timestamp"].unique().sort().to_list()
    if len(stamps) < minimum_observations:
        raise ValueError("insufficient market-model observations")
    if stamps[-1] != fit_end:
        raise ValueError("fit end must equal final observed return close")
    if calendar == "us_equity":
        local = [stamp.astimezone(ZoneInfo("America/New_York")) for stamp in stamps]
        if any(
            stamp.weekday() >= 5 or stamp.timetz().replace(tzinfo=None) not in {time(13), time(16)}
            for stamp in local
        ) or any(b.date() <= a.date() for a, b in zip(local, local[1:])):
            raise ValueError("one return per declared US equity closing session required")
    elif calendar != "crypto_utc" or any(
        (b - a).total_seconds() < 86400 for a, b in zip(stamps, stamps[1:])
    ):
        raise ValueError("one-day model cannot consume intraday returns")
    for symbol in symbols:
        if data.filter(pl.col("symbol") == symbol)["timestamp"].to_list() != stamps:
            raise ValueError("complete common return grid required; no zero filling")
    return data, symbols, stamps


def estimate_market_risk(
    returns: pl.DataFrame,
    *,
    benchmark: str,
    fit_end: datetime,
    model_known_at: datetime,
    currency: str,
    input_sha256: str,
    model_id: str,
    minimum_observations: int = 60,
) -> dict:
    "Fit one market beta with intercept; retain diagonal residual risk, without annualising."
    _model_identity(model_id, currency, input_sha256)
    data, symbols, stamps = _return_panel(
        returns,
        fit_end=fit_end,
        model_known_at=model_known_at,
        minimum_observations=minimum_observations,
    )
    if benchmark not in symbols:
        raise ValueError("explicit market benchmark required in frozen universe")
    x = data.filter(pl.col("symbol") == benchmark)["return"].to_numpy().astype(float)
    centred = x - x.mean()
    xx = float(centred @ centred)
    variance = xx / (len(x) - 1)
    if not math.isfinite(variance) or variance <= 0:
        raise ValueError("market factor must have positive finite sample variance")
    assets, intercepts = [], {}
    for symbol in symbols:
        y = data.filter(pl.col("symbol") == symbol)["return"].to_numpy().astype(float)
        beta = float(centred @ (y - y.mean()) / xx)
        alpha = float(y.mean() - beta * x.mean())
        residual = y - (alpha + beta * x)
        specific = float(residual @ residual / (len(x) - 2))
        if (
            not all(math.isfinite(v) for v in (beta, alpha, specific))
            or abs(beta) > 1e6
            or specific < 0
            or specific > 1e6
            or variance > 1e6
        ):
            raise ValueError("finite bounded market-model estimates required")
        assets.append(
            {"instrument_id": int(symbol), "loadings": [beta], "specific_variance": specific}
        )
        intercepts[symbol] = alpha
    return {
        "model": {
            "model_id": model_id,
            "currency": currency,
            "horizon": "one_day",
            "fit_end": fit_end.isoformat(),
            "model_known_at": model_known_at.isoformat(),
            "factors": ["market"],
            "covariance": [[variance]],
            "assets": assets,
        },
        "fit": {
            "method": "single_market_OLS_with_intercept",
            "benchmarkInstrumentId": int(benchmark),
            "firstReturnAt": stamps[0].isoformat(),
            "lastReturnAt": stamps[-1].isoformat(),
            "observations": len(stamps),
            "minimumObservations": minimum_observations,
            "factorDegreesOfFreedom": 1,
            "residualDegreesOfFreedom": 2,
            "intercepts": intercepts,
            "inputSha256": input_sha256,
            "specificCovariance": "diagonal",
            "annualization": "none",
            "qualificationIssued": False,
            "executionAuthorized": False,
        },
    }


def estimate_style_risk(
    returns: pl.DataFrame,
    exposures: pl.DataFrame,
    *,
    fit_end: datetime,
    model_known_at: datetime,
    currency: str,
    input_sha256: str,
    model_id: str,
    minimum_observations: int = 60,
) -> dict:
    "Retrospective cross-sectional OLS on fixed CURRENT styles; not historical style premia."
    from .factor_risk import _RESERVED, _dated, _numeric

    _model_identity(model_id, currency, input_sha256)
    data, symbols, stamps = _return_panel(
        returns,
        fit_end=fit_end,
        model_known_at=model_known_at,
        minimum_observations=minimum_observations,
        calendar="us_equity",
        maximum_assets=100,
    )
    names = sorted(set(exposures.columns) - {"timestamp", "symbol", "known_at"})
    if (
        not 10 <= len(symbols) <= 100
        or not 1 <= len(names) <= 7
        or "market" in names
        or set(names) & _RESERVED
        or any(not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,31}", name) for name in names)
    ):
        raise ValueError("10 to 100 assets and one to seven distinct current styles required")
    styles = _dated(exposures, {"timestamp", "symbol", "known_at", *names})
    _numeric(styles, names)
    if (
        styles["timestamp"].n_unique() != 1
        or styles.height != len(symbols)
        or styles["symbol"].to_list() != symbols
        or styles.filter(pl.col("timestamp") > model_known_at).height
    ):
        raise ValueError("one complete current exposure snapshot known by model cutoff required")
    X = np.column_stack([np.ones(len(symbols)), styles.select(names).to_numpy().astype(float)])
    if not np.isfinite(X).all() or np.abs(X).max() > 1e6 or np.linalg.matrix_rank(X) != X.shape[1]:
        raise ValueError("full-rank finite bounded style loadings required")
    Y = np.column_stack(
        [
            data.filter(pl.col("symbol") == symbol)["return"].to_numpy().astype(float)
            for symbol in symbols
        ]
    )
    coefficients, _, rank, singular = np.linalg.lstsq(X, Y.T, rcond=None)
    if rank != X.shape[1]:
        raise ValueError("style least-squares rank deficient")
    residual = Y - coefficients.T @ X.T
    F = np.cov(coefficients, ddof=1)
    D = np.var(residual, axis=0, ddof=1)
    if (
        not np.isfinite(F).all()
        or not np.isfinite(D).all()
        or np.abs(F).max() > 1e6
        or np.any(D < 0)
        or np.any(D > 1e6)
    ):
        raise ValueError("finite bounded style covariance and nonnegative specific risk required")
    empirical = np.cov(Y, rowvar=False, ddof=1)
    implied = X @ F @ X.T + np.diag(D)
    denominator = float(np.linalg.norm(empirical, "fro"))
    gap = (
        float(np.linalg.norm(implied - empirical, "fro") / denominator) if denominator > 0 else None
    )
    # Cross-sectional residual orthogonality does not remove time-series cross covariance.
    residual_covariance = np.cov(residual, rowvar=False, ddof=1)
    off_diagonal = residual_covariance - np.diag(D)
    factor_centred = coefficients - coefficients.mean(axis=1, keepdims=True)
    residual_centred = residual - residual.mean(axis=0)
    cross = X @ (factor_centred @ residual_centred / (len(stamps) - 1))
    cross = cross + cross.T
    reconstructed = implied + off_diagonal + cross
    error = empirical - implied
    error_squared = float(np.sum(error * error))
    diagnostics = {
        "method": "exact_sample_covariance_omitted_terms",
        "residualOffDiagonalRelativeNorm": float(np.linalg.norm(off_diagonal, "fro") / denominator)
        if denominator > 0
        else None,
        "factorSpecificCrossRelativeNorm": float(np.linalg.norm(cross, "fro") / denominator)
        if denominator > 0
        else None,
        "residualOffDiagonalSignedGapProjection": float(
            np.sum(off_diagonal * error) / error_squared
        )
        if error_squared > 0
        else None,
        "factorSpecificCrossSignedGapProjection": float(np.sum(cross * error) / error_squared)
        if error_squared > 0
        else None,
        "reconstructionMaxAbsError": float(np.abs(empirical - reconstructed).max()),
        "empiricalMinEigenvalue": float(np.linalg.eigvalsh(empirical).min()),
        "impliedMinEigenvalue": float(np.linalg.eigvalsh(implied).min()),
        "interpretation": "omitted_covariance_terms_not_additive_risk_shares_or_calibration",
    }
    snapshot = styles["timestamp"][0]
    return {
        "model": {
            "model_id": model_id,
            "currency": currency,
            "horizon": "one_day",
            "fit_end": fit_end.isoformat(),
            "model_known_at": model_known_at.isoformat(),
            "factors": ["market", *names],
            "covariance": F.tolist(),
            "assets": [
                {
                    "instrument_id": int(symbol),
                    "loadings": X[i].tolist(),
                    "specific_variance": float(D[i]),
                }
                for i, symbol in enumerate(symbols)
            ],
        },
        "fit": {
            "method": "fixed_current_style_cross_sectional_OLS",
            "calendar": "us_equity",
            "observations": len(stamps),
            "minimumObservations": minimum_observations,
            "assets": len(symbols),
            "firstReturnAt": stamps[0].isoformat(),
            "lastReturnAt": stamps[-1].isoformat(),
            "styleSnapshotAt": snapshot.isoformat(),
            "inputSha256": input_sha256,
            "designRank": int(rank),
            "singularValues": singular.tolist(),
            "crossSectionResidualDegreesOfFreedom": len(symbols) - X.shape[1],
            "sampleCovarianceDegreesOfFreedom": 1,
            "specificCovariance": "diagonal",
            "specificFactorCovariance": "assumed_zero",
            "annualization": "none",
            "relativeEmpiricalCovarianceGap": gap,
            "covarianceDiagnostics": diagnostics,
            "interpretation": "current_fixed_loadings_retrospective_covariance_not_historical_style_premia",
            "qualificationIssued": False,
            "executionAuthorized": False,
        },
    }
