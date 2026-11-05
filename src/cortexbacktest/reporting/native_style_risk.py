"Passive exposures and Euler risk under an explicitly supplied frozen factor model."

from __future__ import annotations

from datetime import datetime
import hashlib
import json
import math
import re

import polars as pl

from .. import strict_json
from .factor_risk import _RESERVED, factor_risk_decomposition, portfolio_style_exposure
from .native_attribution import native_snapshot_weights


def native_style_risk(runs, latest, model):
    fields = {
        "model_id",
        "currency",
        "horizon",
        "fit_end",
        "model_known_at",
        "factors",
        "covariance",
        "assets",
    }
    if not isinstance(model, dict) or set(model) != fields:
        raise ValueError("Explicit complete frozen style/risk model required")
    if (
        not isinstance(model["model_id"], str)
        or not re.fullmatch(r"[A-Za-z0-9._:-]{1,128}", model["model_id"])
        or not isinstance(model["currency"], str)
        or not re.fullmatch(r"[A-Z0-9]{2,8}", model["currency"])
        or model["horizon"] != "one_day"
    ):
        raise ValueError(
            "Explicit model identity, common unit and daily covariance horizon required"
        )
    factors = model["factors"]
    if (
        not isinstance(factors, list)
        or not 1 <= len(factors) <= 8
        or any(
            not isinstance(f, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,31}", f)
            for f in factors
        )
        or len(set(factors)) != len(factors)
        or set(factors) & _RESERVED
    ):
        raise ValueError("Unique bounded nonreserved factor names required")
    n = len(factors)

    def number(value):
        return type(value) in (float, int) and math.isfinite(value) and abs(value) <= 1000000

    matrix = model["covariance"]
    if (
        not isinstance(matrix, list)
        or len(matrix) != n
        or any(
            not isinstance(row, list) or len(row) != n or any(not number(x) for x in row)
            for row in matrix
        )
    ):
        raise ValueError("Explicit finite covariance in declared factor order required")

    fit_end, known = strict_json.clock(model["fit_end"]), strict_json.clock(model["model_known_at"])
    if fit_end > known:
        raise ValueError("Model fit must precede availability")
    symbols = sorted({"INSTRUMENT:" + i for i, _ in latest})
    assets = {}
    if not isinstance(model["assets"], list):
        raise ValueError("Explicit risk assets required")
    for row in model["assets"]:
        if (
            not isinstance(row, dict)
            or set(row) != {"instrument_id", "loadings", "specific_variance"}
            or type(row["instrument_id"]) is not int
            or not isinstance(row["loadings"], list)
            or len(row["loadings"]) != n
            or any(not number(x) for x in row["loadings"])
            or not number(row["specific_variance"])
            or row["specific_variance"] < 0
        ):
            raise ValueError("Complete finite model loadings/specific variances required")
        symbol = "INSTRUMENT:" + str(row["instrument_id"])
        if symbol in assets:
            raise ValueError("Duplicate model asset")
        assets[symbol] = row
    if sorted(assets) != symbols:
        raise ValueError("Model assets must cover the frozen universe exactly")
    weights = {
        name: native_snapshot_weights(runs[name], latest)
        for name in ("qlib", "equal_weight_buy_hold")
    }
    p, b = weights["qlib"], weights["equal_weight_buy_hold"]
    if not p.select("timestamp", "symbol").equals(b.select("timestamp", "symbol")):
        raise ValueError("Portfolio and benchmark exposure snapshots must align exactly")
    first_stamp = p["timestamp"].min()
    if not isinstance(first_stamp, datetime):
        raise ValueError("Native exposure dates must be datetimes")
    if known > first_stamp:
        raise ValueError("Frozen model not known at first exposure snapshot")
    combined = p.join(
        b.select("timestamp", "symbol", pl.col("weight").alias("benchmark_weight")),
        on=["timestamp", "symbol"],
    )
    assets["__CASH__"] = {"loadings": [0.0] * n, "specific_variance": 0.0}
    exposure = pl.DataFrame(
        [
            {
                "timestamp": row["timestamp"],
                "symbol": row["symbol"],
                "known_at": known,
                **dict(zip(factors, assets[row["symbol"]]["loadings"], strict=True)),
            }
            for row in combined.iter_rows(named=True)
        ]
    )
    style = portfolio_style_exposure(combined, exposure, model_id=model["model_id"]).to_dict()
    stamp = p["timestamp"].max()
    if not isinstance(stamp, datetime):
        raise ValueError("Native risk as-of must be a datetime")
    current = exposure.filter(pl.col("timestamp") == stamp)
    specific = current.select("timestamp", "symbol", "known_at").with_columns(
        pl.Series("variance", [assets[s]["specific_variance"] for s in current["symbol"]])
    )
    covariance = pl.DataFrame(
        [{"factor": f, **dict(zip(factors, matrix[i], strict=True))} for i, f in enumerate(factors)]
    )
    risk = {}
    for name in ("portfolio", "benchmark", "active"):
        if name == "active":
            w = combined.filter(pl.col("timestamp") == stamp).select(
                "timestamp",
                "symbol",
                "known_at",
                (pl.col("weight") - pl.col("benchmark_weight")).alias("weight"),
            )
        else:
            w = weights["qlib" if name == "portfolio" else "equal_weight_buy_hold"].filter(
                pl.col("timestamp") == stamp
            )
        risk[name] = factor_risk_decomposition(
            w,
            current,
            covariance,
            specific,
            as_of=stamp,
            model_known_at=known,
            fit_end=fit_end,
            model_id=model["model_id"],
            currency=model["currency"],
            horizon=model["horizon"],
        )
    return {
        "modelId": model["model_id"],
        "style": style,
        "risk": risk,
        "snapshotCount": p["timestamp"].n_unique(),
        "modelSha256": hashlib.sha256(
            json.dumps(model, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest(),
        "evidence": "caller_supplied_frozen_prestandardized_loadings_not_authenticated_PIT_or_estimated_styles",
        "driftBasis": "actual_holdings_change_under_one_frozen_model_not_factor_or_regime_refitting",
        "cashAssumption": "explicit_zero_factor_loadings_and_specific_variance_simulation_assumption",
        "specificRiskAssumption": "diagonal_specific_covariance",
        "riskSnapshot": "last_native_equity_close_portfolio_benchmark_and_active",
        "annualization": "none_daily_covariance_and_volatility",
        "qualificationIssued": False,
        "executionAuthorized": False,
    }
