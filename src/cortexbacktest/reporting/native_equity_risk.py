"Fit current same-USD equity styles; adjusted-close analytics never grants order admission."

from datetime import UTC, datetime, timedelta
import hashlib
import json

import polars as pl

from ..adapters.equity_risk import load_equity_risk_inputs
from .factor_risk import (
    factor_risk_decomposition,
    portfolio_style_exposure,
    standardise_style_exposures,
)
from .risk_model import estimate_style_risk


def fit_equity_risk_bundle(path, *, manifest_sha256, model_known_at=None, max_fit_age_days=7, retrospective=False,
                          prospective=None, forecast_sessions=None):
    inputs = load_equity_risk_inputs(path, sha256=manifest_sha256)
    manifest, characteristics, returns = inputs.manifest, inputs.characteristics, inputs.returns
    admission = inputs.execution_admission
    cutoff = datetime.fromisoformat(manifest["asOf"])
    definitions = {"size": {"log_market_cap": 1.0}, "value": {"book_to_market": 1.0}}
    styles = standardise_style_exposures(characteristics, definitions=definitions)
    end = returns["timestamp"].max()
    if not isinstance(end, datetime):
        raise ValueError("explicit observed equity fit end required")
    fitted = estimate_style_risk(
        returns,
        styles,
        fit_end=end,
        model_known_at=cutoff,
        currency="USD",
        input_sha256=manifest_sha256,
        model_id="native-current-equity-style-v1",
    )
    built = datetime.now(UTC) if model_known_at is None else model_known_at
    if (
        not isinstance(built, datetime)
        or built.tzinfo is None
        or built.utcoffset() is None
        or built < cutoff
        or built > datetime.now(UTC)
    ):
        raise ValueError("model production clock must follow received input cutoff")
    if type(max_fit_age_days) is not int or not 1 <= max_fit_age_days <= 30:
        raise ValueError("bounded maximum equity fit age required")
    if type(retrospective) is not bool:
        raise ValueError("explicit retrospective flag required")
    freshness_at = cutoff if retrospective else built
    if freshness_at > inputs.style_expires_at or freshness_at - end > timedelta(days=max_fit_age_days):
        raise ValueError("current equity model inputs stale at production time")
    model = fitted["model"]
    model["model_known_at"] = built.isoformat()
    fitted["fit"].update(
        {
            "inputCutoff": cutoff.isoformat(),
            "computedAt": built.isoformat(),
            "styleInputsSha256": manifest["styleInputSha256"],
            "priceInputsSha256": manifest["priceInputSha256"],
            "returnBasis": manifest["priceBasis"],
        }
    )
    assets, factors = model["assets"], model["factors"]
    supplied_weights = inputs.portfolio_weights
    caps = dict(characteristics.select("symbol", "market_cap").iter_rows())
    total = sum(caps.values())
    base = [
        {"timestamp": built, "known_at": built, "symbol": "INSTRUMENT:" + str(a["instrument_id"])}
        for a in assets
    ]
    combined = pl.DataFrame(
        [
            {
                **b,
                "weight": (supplied_weights["weights"]["portfolio"][b["symbol"]]
                           if supplied_weights else 1 / len(assets)),
                "benchmark_weight": (supplied_weights["weights"]["benchmark"][b["symbol"]]
                                     if supplied_weights else caps[str(a["instrument_id"])] / total),
            }
            for b, a in zip(base, assets, strict=True)
        ]
    )
    exposure = pl.DataFrame(
        [
            {**b, **dict(zip(factors, a["loadings"], strict=True))}
            for b, a in zip(base, assets, strict=True)
        ]
    )
    specific = pl.DataFrame(
        [{**b, "variance": a["specific_variance"]} for b, a in zip(base, assets, strict=True)]
    )
    if supplied_weights:
        balances = ["__CASH__"]
        if "__DIVIDEND_RECEIVABLE__" in supplied_weights["weights"]["portfolio"]:
            balances.append("__DIVIDEND_RECEIVABLE__")
        for symbol in balances:
            balance = {"timestamp": built, "known_at": built, "symbol": symbol}
            combined = combined.vstack(pl.DataFrame([{
                **balance, "weight": supplied_weights["weights"]["portfolio"][symbol],
                "benchmark_weight": supplied_weights["weights"]["benchmark"][symbol],
            }]))
            exposure = exposure.vstack(pl.DataFrame([{**balance, **dict.fromkeys(factors, 0.0)}]))
            specific = specific.vstack(pl.DataFrame([{**balance, "variance": 0.0}]))
    covariance = pl.DataFrame(
        [
            {"factor": f, **dict(zip(factors, model["covariance"][i], strict=True))}
            for i, f in enumerate(factors)
        ]
    )
    risk = {}
    for name, value in [
        ("portfolio", pl.col("weight")),
        ("benchmark", pl.col("benchmark_weight")),
        ("active", pl.col("weight") - pl.col("benchmark_weight")),
    ]:
        weights = combined.select("timestamp", "symbol", "known_at", value.alias("weight"))
        risk[name] = factor_risk_decomposition(
            weights,
            exposure,
            covariance,
            specific,
            as_of=built,
            model_known_at=built,
            fit_end=end,
            model_id=model["model_id"],
            currency="USD",
            horizon="one_day",
        )
    report = {
        "status": "available",
        "analysisMode": "retrospective_recomputation" if retrospective else "current_snapshot",
        "model": model,
        "fit": fitted["fit"],
        "risk": risk,
        "style": portfolio_style_exposure(combined, exposure, model_id=model["model_id"]).to_dict(),
        "modelSha256": hashlib.sha256(
            json.dumps(model, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest(),
        "inputSha256": manifest_sha256,
        "asOf": built.isoformat(),
        "inputAsOf": cutoff.isoformat(),
        "styleSnapshotAt": manifest["styleSnapshotAt"],
        "currency": "USD",
        "portfolioDefinition": ("owned_native_research_closing_holdings_not_live_account" if supplied_weights
                                else "declared_equal_weight_validation_portfolio_not_live_account"),
        "benchmarkDefinition": ("source_native_initial_equal_weight_buy_hold_closing_holdings" if supplied_weights
                                else "current_scope_cap_weights_not_market_index"),
        "portfolioSource": supplied_weights,
        "cashRiskAssumption": "zero_factor_loadings_and_specific_variance_research_assumption" if supplied_weights else None,
        "methodDefinition": {
            "version": 1,
            "size": "positive_standardized_log_market_cap_larger_means_larger_cap",
            "value": "positive_standardized_book_to_market",
            "estimator": "equal_weight_cross_sectional_ols_fixed_current_exposures",
            "riskHorizon": "one_day_not_annualized",
            "specificCovariance": "diagonal",
        },
        "styleInterpretation": "fixed_current_style_cross_sectional_fit_not_historical_style_portfolios_or_forecasts",
        "identityEvidence": inputs.identity_evidence,
        "executionAdmission": admission,
        "executionAdmitted": False,
        "historicalIdentityCertified": False,
        "historicalInputSemantics": {
            "certified": False,
            "identity": "current_received_issuer_snapshot_not_continuous_historical_security",
            "currency": "current_USD_evidence_not_historical_currency_certification",
            "shareClass": "historical_share_class_evidence_unavailable",
            "adjustment": "received_vendor_total_return_provenance_not_certified_action_coverage",
            "marketCapitalisation": "current_profile_not_historical_market_cap_basis",
        },
        "qualificationIssued": False,
        "executionAuthorized": False,
    }
    if prospective is not None or forecast_sessions is not None:
        if retrospective or prospective is None or forecast_sessions is None:
            raise ValueError("only a new current model can freeze prospective forecasts")
        from .equity_risk_forward import freeze_equity_forecast

        report["forecast"] = freeze_equity_forecast(report, returns, combined, prospective, forecast_sessions)
    return report
