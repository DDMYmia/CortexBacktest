"VaR exception calibration; no forecast fitting or trading qualification."
from __future__ import annotations

from dataclasses import dataclass
import math

import polars as pl


def _probability(value: float, name: str) -> None:
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value < 1:
        raise ValueError(f"{name} must be finite and in (0, 1)")


def _likelihood(failures: int, successes: int, probability: float) -> float:
    value = 0.
    if failures:
        value += failures * math.log1p(-probability) if probability < 1 else -math.inf
    if successes:
        value += successes * math.log(probability) if probability > 0 else -math.inf
    return value


def kupiec_pof(violations: int, observations: int, *, confidence: float = .99) -> dict:
    _probability(confidence, "confidence")
    if (type(observations) is not int or observations < 1 or type(violations) is not int
            or not 0 <= violations <= observations):
        raise ValueError("integer exception count must be within positive observations")
    realized = violations / observations
    fitted = _likelihood(observations - violations, violations, realized)
    null = (observations - violations) * math.log(confidence) + violations * math.log1p(-confidence)
    statistic = max(0., 2 * (fitted - null))
    return {"statistic": statistic, "pValue": math.erfc(math.sqrt(statistic / 2)),
            "degreesOfFreedom": 1, "expectedViolations": observations * (1 - confidence),
            "realizedViolationRate": realized, "method": "kupiec_unconditional_coverage"}


def christoffersen_independence(breaches: list[bool]) -> dict:
    if len(breaches) < 2 or any(type(x) is not bool for x in breaches):
        raise ValueError("need at least two boolean breach observations")
    counts = {"n00": 0, "n01": 0, "n10": 0, "n11": 0}
    for a, b in zip(breaches[:-1], breaches[1:], strict=True):
        counts[f"n{int(a)}{int(b)}"] += 1
    n00, n01, n10, n11 = (counts[k] for k in ("n00", "n01", "n10", "n11"))
    result = {"statistic": None, "pValue": None, "degreesOfFreedom": 1,
              "transitions": counts, "reason": None}
    if not n00 + n01 or not n10 + n11:
        return {**result, "reason": "unobserved_transition_origin"}
    p0, p1 = n01 / (n00 + n01), n11 / (n10 + n11)
    pooled = (n01 + n11) / (len(breaches) - 1)
    null = _likelihood(n00 + n10, n01 + n11, pooled)
    fitted = _likelihood(n00, n01, p0) + _likelihood(n10, n11, p1)
    statistic = max(0., 2 * (fitted - null))
    return {**result, "statistic": statistic, "pValue": math.erfc(math.sqrt(statistic / 2))}


@dataclass(frozen=True, slots=True)
class VarBacktestResult:
    observations: pl.DataFrame
    summary: dict

    def to_dict(self) -> dict:
        rows = self.observations.with_columns(
            pl.col("timestamp", "forecast_at", "period_start").dt.to_string("%+")
        ).to_dicts()
        return {"summary": self.summary, "observations": rows,
                "qualificationIssued": False, "executionAuthorized": False}


def var_backtest(
    returns: pl.DataFrame, forecasts: pl.DataFrame, *, confidence: float = .99,
    significance: float = .05,
) -> VarBacktestResult:
    "Exactly align dated returns and positive-loss VaR forecasts known before each return."
    _probability(confidence, "confidence")
    _probability(significance, "significance")
    for frame, columns in ((returns, {"timestamp", "return", "period_start"}),
                           (forecasts, {"timestamp", "var", "forecast_at"})):
        if set(frame.columns) != columns or not frame.height:
            raise ValueError("explicit nonempty return/forecast columns required")
        for key in columns & {"timestamp", "forecast_at", "period_start"}:
            dtype = frame.schema[key]
            if not isinstance(dtype, pl.Datetime) or not dtype.time_zone or frame[key].null_count():
                raise ValueError("aware non-null observation and forecast clocks required")
        if frame["timestamp"].n_unique() != frame.height:
            raise ValueError("duplicate risk observation timestamp")
    for frame, field in ((returns, "return"), (forecasts, "var")):
        if not frame.schema[field].is_numeric() or frame[field].null_count() or not frame[field].is_finite().all():
            raise ValueError("finite complete returns and VaR required")
    left = returns.with_columns(pl.col("timestamp").dt.convert_time_zone("UTC").dt.cast_time_unit("ns"),
                                pl.col("period_start").dt.convert_time_zone("UTC").dt.cast_time_unit("ns"))
    right = forecasts.with_columns(pl.col("timestamp").dt.convert_time_zone("UTC").dt.cast_time_unit("ns"),
                                   pl.col("forecast_at").dt.convert_time_zone("UTC").dt.cast_time_unit("ns"))
    if (left.height != right.height or left.join(right, on="timestamp", how="anti").height
            or right.join(left, on="timestamp", how="anti").height):
        raise ValueError("VaR forecasts must cover exactly the same return timestamps")
    frame = left.join(right, on="timestamp").sort("timestamp")
    if frame.filter((pl.col("var") < 0) | (pl.col("forecast_at") > pl.col("period_start"))
                    | (pl.col("period_start") >= pl.col("timestamp"))).height:
        raise ValueError("VaR must be nonnegative and known by the start of a positive return interval")
    if frame.height > 1 and frame.with_columns(pl.col("timestamp").shift(1).alias("prior_end")).slice(1).filter(pl.col("period_start") != pl.col("prior_end")).height > 0:
        raise ValueError("return intervals must be contiguous and nonoverlapping")
    frame = frame.with_columns((pl.col("return") < -pl.col("var")).alias("breach"))
    breaches = frame["breach"].to_list()
    unconditional = kupiec_pof(sum(breaches), len(breaches), confidence=confidence)
    independence = (christoffersen_independence(breaches) if len(breaches) > 1 else
        {"statistic": None, "pValue": None, "reason": "insufficient_observations"})
    conditional = {"statistic": None, "pValue": None, "degreesOfFreedom": 2,
                   "reason": independence["reason"]}
    if independence["statistic"] is not None:
        statistic = unconditional["statistic"] + independence["statistic"]
        conditional.update(statistic=statistic, pValue=math.exp(-statistic / 2), reason=None)
    for result in (unconditional, independence, conditional):
        result["rejectNull"] = result["pValue"] < significance if result["pValue"] is not None else None
    return VarBacktestResult(frame, {"confidence": confidence, "significance": significance,
        "observations": len(breaches), "violations": sum(breaches), "unconditionalCoverage": unconditional,
        "independence": independence, "conditionalCoverage": conditional,
        "breachPolicy": "return_strictly_less_than_negative_var",
        "timingPolicy": "forecast_at_at_or_before_period_start_strictly_before_return_timestamp",
        "interpretation": "asymptotic_calibration_diagnostic_not_risk_or_trading_approval"})


def historical_var_calibration(equity: pl.DataFrame, *, window: int, confidence: float,
                               significance: float) -> dict:
    "Calibrate rolling empirical VaR against subsequent native equity intervals."
    import numpy as np

    if type(window) is not int or not 20 <= window <= 1000:
        raise ValueError("VaR window must be an integer in 20..1000")
    _probability(confidence, "confidence")
    _probability(significance, "significance")
    if not .8 <= confidence <= .9999 or not .001 <= significance <= .2:
        raise ValueError("Explicit supported risk confidence/significance required")
    if set(equity.columns) != {"timestamp", "equity"} or not equity.height:
        raise ValueError("Explicit nonempty native timestamp/equity columns required")
    clock = equity.schema["timestamp"]
    if (not isinstance(clock, pl.Datetime) or not clock.time_zone
            or equity["timestamp"].null_count() or equity["timestamp"].n_unique() != equity.height
            or not equity.schema["equity"].is_numeric() or equity["equity"].null_count()
            or not equity["equity"].is_finite().all() or (equity["equity"] <= 0).any()):
        raise ValueError("Unique aware timestamps and complete finite positive equity required")
    data = equity.sort("timestamp")
    values = data["equity"].to_numpy().astype(float)
    realized = values[1:] / values[:-1] - 1
    if not np.isfinite(realized).all():
        raise ValueError("Native equity interval returns must be finite")
    available = max(0, len(realized) - window)
    warnings = []
    if window * (1 - confidence) < 5:
        warnings.append("fewer_than_five_expected_tail_samples_in_forecast_window")
    if available * (1 - confidence) < 5:
        warnings.append("fewer_than_five_expected_calibration_breaches")
    result = {"status": "unavailable", "reason": "insufficient_post_warmup_observations",
        "totalReturnIntervals": len(realized), "warmupExcluded": min(window, len(realized)),
        "availableForecastIntervals": available, "calibratedIntervals": 0,
        "minimumCalibrationIntervals": 30, "warnings": warnings, "summary": None,
        "observations": [], "qualificationIssued": False, "executionAuthorized": False}
    if available < 30:
        return result
    forecasts = [max(0., -float(np.quantile(realized[i-window:i], 1-confidence, method="linear")))
                 for i in range(window, len(realized))]
    observations = pl.DataFrame({"timestamp": data["timestamp"].slice(window+1),
        "period_start": data["timestamp"].slice(window, available), "return": realized[window:]})
    forecast = pl.DataFrame({"timestamp": observations["timestamp"],
        "forecast_at": observations["period_start"], "var": forecasts})
    calibrated = var_backtest(observations, forecast, confidence=confidence,
                              significance=significance).to_dict()
    return {**result, **calibrated, "status": "available", "reason": None,
            "calibratedIntervals": available}
