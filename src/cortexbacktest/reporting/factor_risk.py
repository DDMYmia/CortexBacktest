"Complete named style exposures, drift and Euler factor-risk reports."

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
import math

import numpy as np
import polars as pl


_KEYS = {"timestamp", "symbol", "known_at"}
_RESERVED = {*_KEYS, "weight", "benchmark_weight", "variance", "factor"}


def _named(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"explicit {name} required")


def _numeric(frame: pl.DataFrame, names) -> None:
    for name in names:
        if (
            not frame.schema[name].is_numeric()
            or frame[name].null_count()
            or not frame[name].is_finite().all()
        ):
            raise ValueError(
                "complete finite factor-model inputs required; no implicit zero filling"
            )


def _dated(frame: pl.DataFrame, columns: set[str]) -> pl.DataFrame:
    if set(frame.columns) != columns or not frame.height:
        raise ValueError("explicit nonempty factor-model columns required")
    for name in ("timestamp", "known_at"):
        dtype = frame.schema[name]
        if not isinstance(dtype, pl.Datetime) or not dtype.time_zone or frame[name].null_count():
            raise ValueError("aware complete factor-model clocks required")
    if (
        frame.schema["symbol"] != pl.String
        or frame["symbol"].null_count()
        or any(not x.strip() or x != x.strip() for x in frame["symbol"])
    ):
        raise ValueError("explicit factor-model symbols required")
    result = frame.with_columns(
        pl.col("timestamp", "known_at").dt.convert_time_zone("UTC").dt.cast_time_unit("ns")
    )
    if result.select("timestamp", "symbol").unique().height != result.height:
        raise ValueError("duplicate factor-model timestamp/symbol")
    if result.filter(pl.col("known_at") > pl.col("timestamp")).height:
        raise ValueError("factor-model inputs must be known by exposure timestamp")
    return result.sort("timestamp", "symbol")


def standardise_style_exposures(
    characteristics: pl.DataFrame,
    *,
    definitions: Mapping[str, Mapping[str, float]],
    min_assets: int = 10,
    winsorise: float = 0.025,
) -> pl.DataFrame:
    "Build declared composite styles: winsorise, cap-centre, equal-weight sample-standardise."
    if (
        type(min_assets) is not int
        or min_assets < 3
        or not math.isfinite(winsorise)
        or not 0 <= winsorise < 0.5
        or not definitions
    ):
        raise ValueError("explicit style definitions and meaningful cross-section required")
    factors, fields = sorted(definitions), set()
    for factor, components in definitions.items():
        _named(factor, "style factor")
        if factor in _RESERVED or not components:
            raise ValueError("factor name collides with metadata or has no components")
        for name, coefficient in components.items():
            _named(name, "characteristic")
            if (
                name in {*_KEYS, "market_cap"}
                or type(coefficient) not in (int, float)
                or not math.isfinite(coefficient)
                or not coefficient
            ):
                raise ValueError("explicit finite nonzero characteristic coefficient required")
            fields.add(name)
    data = _dated(characteristics, {*_KEYS, "market_cap", *fields})
    _numeric(data, ["market_cap", *fields])
    if data.filter(pl.col("market_cap") <= 0).height:
        raise ValueError("positive complete market caps required")
    results = []
    for piece in data.partition_by("timestamp", maintain_order=True):
        if piece.height < min_assets:
            raise ValueError("insufficient cross-section for declared style model")
        caps = piece["market_cap"].to_numpy().astype(float)
        cap_weights = caps / caps.max()
        cap_weights /= cap_weights.sum()
        standard = {}
        for name in fields:
            values = piece[name].to_numpy().astype(float)
            low, high = np.quantile(values, [winsorise, 1 - winsorise])
            clipped = np.clip(values, low, high)
            spread = float(np.std(clipped, ddof=1))
            if low == high or not math.isfinite(spread) or spread <= 0:
                raise ValueError("constant or unrepresentable characteristic cannot define a style")
            standard[name] = (clipped - cap_weights @ clipped) / spread
        result = piece.select("timestamp", "symbol", "known_at").with_columns(
            pl.col("known_at").max().alias("known_at")
        )
        for factor in factors:
            components = definitions[factor]
            exposure = np.zeros(piece.height, dtype=float)
            for name, coefficient in components.items():
                exposure += coefficient * standard[name]
            exposure /= math.fsum(abs(x) for x in components.values())
            if not np.isfinite(exposure).all():
                raise ValueError("unrepresentable standardized style exposure")
            result = result.with_columns(pl.Series(factor, exposure))
        results.append(result)
    return pl.concat(results)


@dataclass(frozen=True, slots=True)
class StyleExposureReport:
    summary: dict
    history: pl.DataFrame
    drift: pl.DataFrame

    def to_dict(self) -> dict:
        return {
            "summary": self.summary,
            "history": self.history.with_columns(pl.col("timestamp").dt.to_string("%+")).to_dicts(),
            "drift": self.drift.to_dicts(),
            "qualificationIssued": False,
            "executionAuthorized": False,
        }


def portfolio_style_exposure(
    weights: pl.DataFrame,
    exposures: pl.DataFrame,
    *,
    model_id: str,
) -> StyleExposureReport:
    "Aggregate named complete exposures; a missing held asset is not a market-average holding."
    _named(model_id, "model identity/definition")
    factors = sorted(set(exposures.columns) - _KEYS)
    if not factors or any(f in _RESERVED or not f.strip() or f != f.strip() for f in factors):
        raise ValueError("nonreserved named style factors required")
    data = _dated(exposures, {*_KEYS, *factors})
    holdings = _dated(weights, {*_KEYS, "weight", "benchmark_weight"})
    _numeric(data, factors)
    _numeric(holdings, ["weight", "benchmark_weight"])
    keys = ["timestamp", "symbol"]
    if (
        holdings.join(data, on=keys, how="anti").height
        or data.join(holdings, on=keys, how="anti").height
    ):
        raise ValueError(
            "explicit exposures and weights must cover exactly the same timestamp/symbol universe"
        )
    joined = holdings.join(data.drop("known_at"), on=keys)
    rows = []
    for piece in joined.partition_by("timestamp", maintain_order=True):
        stamp = piece["timestamp"].cast(pl.Int64)[0]
        for name in ["weight", "benchmark_weight"]:
            if not math.isclose(math.fsum(piece[name]), 1.0, abs_tol=1e-10, rel_tol=0):
                raise ValueError("style holdings must include all funded assets and cash")
        for factor in factors:
            p = math.fsum(w * x for w, x in zip(piece["weight"], piece[factor], strict=True))
            b = math.fsum(
                w * x for w, x in zip(piece["benchmark_weight"], piece[factor], strict=True)
            )
            if not all(math.isfinite(x) for x in [p, b, p - b]):
                raise ValueError("unrepresentable portfolio style exposure")
            rows.append(
                {
                    "timestamp": stamp,
                    "factor": factor,
                    "portfolio_exposure": p,
                    "benchmark_exposure": b,
                    "active_exposure": p - b,
                }
            )
    history = pl.DataFrame(rows).with_columns(pl.col("timestamp").cast(pl.Datetime("ns", "UTC")))
    drift = []
    for factor in factors:
        values = history.filter(pl.col("factor") == factor)["active_exposure"].to_list()
        mean = math.fsum(values) / len(values)
        deviation = (
            math.sqrt(math.fsum((x - mean) ** 2 for x in values) / (len(values) - 1))
            if len(values) > 1
            else None
        )
        drift.append(
            {
                "factor": factor,
                "observations": len(values),
                "mean_active_exposure": mean,
                "std_active_exposure": deviation,
                "first_active_exposure": values[0],
                "last_active_exposure": values[-1],
                "total_change": values[-1] - values[0],
                "max_abs_step": max(
                    (abs(b - a) for a, b in zip(values[:-1], values[1:], strict=True)), default=None
                ),
            }
        )
    if any(isinstance(v, float) and not math.isfinite(v) for row in drift for v in row.values()):
        raise ValueError("unrepresentable style drift statistic")
    return StyleExposureReport(
        {
            "modelId": model_id,
            "coverage": "complete_explicit_universe_no_imputation",
            "driftBasis": "active_exposure_same_declared_model_not_a_statistical_significance_test",
            "evidence": "caller_supplied_known_at_not_authenticated_fundamental_PIT",
        },
        history,
        pl.DataFrame(drift),
    )


def factor_risk_decomposition(
    weights: pl.DataFrame,
    exposures: pl.DataFrame,
    covariance: pl.DataFrame,
    specific_variances: pl.DataFrame,
    *,
    as_of: datetime,
    model_known_at: datetime,
    fit_end: datetime,
    model_id: str,
    currency: str,
    horizon: str,
) -> dict:
    "Euler decomposition using one complete dated model; D is explicitly diagonal."
    for value, name in [
        (model_id, "model identity"),
        (currency, "currency"),
        (horizon, "covariance horizon"),
    ]:
        _named(value, name)
    for clock in [as_of, model_known_at, fit_end]:
        if not isinstance(clock, datetime) or clock.tzinfo is None or clock.utcoffset() is None:
            raise ValueError("aware model fit/availability/as-of clocks required")
    if not fit_end <= model_known_at <= as_of:
        raise ValueError("model must be fitted and available by risk as-of")
    factors = sorted(set(exposures.columns) - _KEYS)
    if not factors or any(f in _RESERVED or not f.strip() or f != f.strip() for f in factors):
        raise ValueError("nonreserved named risk factors required")
    x = _dated(exposures, {*_KEYS, *factors})
    w = _dated(weights, {*_KEYS, "weight"})
    d = _dated(specific_variances, {*_KEYS, "variance"})
    for frame, names in [(x, factors), (w, ["weight"]), (d, ["variance"])]:
        _numeric(frame, names)
        if frame.filter(pl.col("timestamp") != as_of).height:
            raise ValueError("risk inputs must cover exactly the declared as-of timestamp")
    symbols = w["symbol"].to_list()
    if x["symbol"].to_list() != symbols or d["symbol"].to_list() != symbols:
        raise ValueError("every risk asset needs explicit weights, exposures and specific variance")
    if d.filter(pl.col("variance") < 0).height:
        raise ValueError("specific variances cannot be negative or silently clipped")
    if (
        set(covariance.columns) != {"factor", *factors}
        or covariance.height != len(factors)
        or covariance.schema["factor"] != pl.String
        or covariance["factor"].null_count()
        or set(covariance["factor"]) != set(factors)
    ):
        raise ValueError("covariance must cover exactly all named factors")
    _numeric(covariance, factors)
    F = covariance.sort("factor").select(factors).to_numpy().astype(float)
    scale = max(float(np.abs(F).max()), np.finfo(float).tiny)
    if not np.allclose(F, F.T, rtol=0, atol=scale * 1e-12):
        raise ValueError("factor covariance must be symmetric")
    original = F
    F = (F + F.T) / 2
    eigenvalues, eigenvectors = np.linalg.eigh(F)
    if eigenvalues.min() < -scale * 1e-12:
        raise ValueError("factor covariance must be positive semidefinite")
    if eigenvalues.min() < 0:
        F = (eigenvectors * np.maximum(eigenvalues, 0)) @ eigenvectors.T
    X = x.select(factors).to_numpy().astype(float)
    W = w["weight"].to_numpy().astype(float)
    D = d["variance"].to_numpy().astype(float)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        try:
            b = X.T @ W
            Fb = F @ b
            factor_var = float(b @ Fb)
            specific_var = float((W * W) @ D)
            if factor_var < 0 and abs(factor_var) <= scale * max(1.0, float(b @ b)) * 1e-12:
                factor_var = 0.0
            total = factor_var + specific_var
            if not math.isfinite(total) or total < 0:
                raise ValueError("unrepresentable factor-model variance")
            vol = math.sqrt(total)
            factor_rc = b * Fb / vol if vol else np.zeros(len(factors))
            specific_rc = W * W * D / vol if vol else np.zeros(len(symbols))
            asset_rc = W * (X @ Fb + D * W) / vol if vol else np.zeros(len(symbols))
        except FloatingPointError as error:
            raise ValueError("unrepresentable factor-model risk arithmetic") from error
    residual = vol - math.fsum([*factor_rc, *specific_rc])
    if not math.isclose(math.fsum(asset_rc), vol, rel_tol=1e-9, abs_tol=1e-12) or abs(
        residual
    ) > max(1e-12, vol * 1e-9):
        raise ValueError("Euler risk components failed reconciliation")
    return {
        "modelId": model_id,
        "currency": currency,
        "horizon": horizon,
        "asOf": as_of.isoformat(),
        "modelKnownAt": model_known_at.isoformat(),
        "fitEnd": fit_end.isoformat(),
        "totalVariance": total,
        "factorVariance": factor_var,
        "specificVariance": specific_var,
        "volatility": vol,
        "riskShareStatus": "available" if vol else "undefined_zero_risk",
        "factors": [
            {
                "factor": name,
                "exposure": float(b[i]),
                "volatilityContribution": float(factor_rc[i]),
                "riskShare": float(factor_rc[i] / vol) if vol else None,
            }
            for i, name in enumerate(factors)
        ],
        "assets": [
            {
                "symbol": name,
                "volatilityContribution": float(asset_rc[i]),
                "specificVolatilityContribution": float(specific_rc[i]),
                "riskShare": float(asset_rc[i] / vol) if vol else None,
            }
            for i, name in enumerate(symbols)
        ],
        "reconciliationResidual": residual,
        "covarianceRoundoffAdjustment": float(np.abs(F - original).max()),
        "assumptions": "supplied_factor_model_diagonal_specific_covariance_no_estimation_or_automatic_annualization",
        "evidence": "caller_supplied_model_clocks_not_authenticated_PIT_risk_model",
        "qualificationIssued": False,
        "executionAuthorized": False,
    }
