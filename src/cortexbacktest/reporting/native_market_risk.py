"Received-cutoff risk fit for frozen research holdings; never a historical risk forecast."

from datetime import UTC, datetime, timedelta
import hashlib
import json

import polars as pl

from .factor_risk import factor_risk_decomposition
from .native_attribution import native_snapshot_weights
from .risk_model import estimate_market_risk
from .. import strict_json


def validate_quote_evidence(document, *, instruments, currency, as_of):
    if (
        set(document) != {"contractVersion", "asOf", "currency", "records"}
        or document["contractVersion"] != "market-quote-evidence-v1"
        or strict_json.clock(document["asOf"]) != as_of
        or document["currency"] != currency
    ):
        raise ValueError("quote evidence scope mismatch")
    identifiers = [r["instrumentId"] for r in document["records"]]
    if (
        any(type(i) is not int or i <= 0 for i in identifiers)
        or len(set(identifiers)) != len(identifiers)
        or set(identifiers) != set(instruments)
    ):
        raise ValueError("quote evidence must exactly cover frozen instruments")
    for row in document["records"]:
        receipt_bytes, raw = row["receiptUtf8"].encode(), row["rawUtf8"].encode()
        if hashlib.sha256(receipt_bytes).hexdigest() != row["receiptSha256"]:
            raise ValueError("exchange receipt digest mismatch")
        receipt = strict_json.loads(receipt_bytes)
        if (
            receipt["contractVersion"] != "research-source-receipt-v1"
            or receipt["rawFile"] != "raw.data"
            or receipt["rawBytes"] != len(raw)
            or hashlib.sha256(raw).hexdigest() != receipt["rawSha256"]
            or receipt["source"] != "binance"
            or receipt["dataset"] != "exchange_info"
            or receipt["status"] != "received"
            or receipt["httpStatus"] != 200
            or not strict_json.clock(receipt["receivedAt"])
            == strict_json.clock(row["receivedAt"])
            <= strict_json.clock(row["recordedAt"])
            <= as_of
        ):
            raise ValueError("received exchange metadata source/clock mismatch")
        payload = strict_json.loads(raw)
        matching = [r for r in payload["symbols"] if r["symbol"] == row["symbol"]]
        if len(matching) != 1:
            raise ValueError("one exchange identity per frozen asset required")
        asset = matching[0]
        base, quote = asset["baseAsset"], asset["quoteAsset"]
        if (
            not all(
                isinstance(v, str) and v.isascii() and v.isalnum() and v == v.upper()
                for v in (base, quote)
            )
            or base + quote != row["symbol"]
            or quote != currency
            or asset["status"] != "TRADING"
            or asset["isSpotTradingAllowed"] is not True
        ):
            raise ValueError("verified common spot quote currency required")


def native_market_risk(runs, latest, config, *, as_of, input_sha256, quotes, model_known_at=None):
    if (
        not isinstance(config, dict)
        or set(config) != {"benchmark_instrument_id", "window", "max_fit_age_days", "currency"}
        or type(config["window"]) is not int
        or not 60 <= config["window"] <= 252
        or type(config["max_fit_age_days"]) is not int
        or not 1 <= config["max_fit_age_days"] <= 30
        or type(config["benchmark_instrument_id"]) is not int
    ):
        raise ValueError("explicit bounded native market risk options required")
    ids = sorted({int(r["instrumentId"]) for r in latest.values()})
    if config["benchmark_instrument_id"] not in ids:
        raise ValueError("market benchmark must be in frozen universe")
    validate_quote_evidence(quotes, instruments=ids, currency=config["currency"], as_of=as_of)
    panel = {strict_json.clock(r["sessionClose"]): {} for r in latest.values()}
    for row in latest.values():
        panel[strict_json.clock(row["sessionClose"])][int(row["instrumentId"])] = row
    stamps = sorted(panel)[-(config["window"] + 1) :]
    base_report = {
        "config": config,
        "asOf": as_of.isoformat(),
        "inputAsOf": as_of.isoformat(),
        "inputSha256": input_sha256,
        "quoteEvidenceSha256": hashlib.sha256(
            json.dumps(quotes, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "interpretation": "input_cutoff_diagnostic_for_frozen_research_holdings_not_historical_forecast_or_live_account",
        "qualificationIssued": False,
        "executionAuthorized": False,
    }
    if len(stamps) < config["window"] + 1:
        return {**base_report, "status": "unavailable", "reason": "insufficient_complete_history"}
    end = stamps[-1]
    if end > as_of:
        raise ValueError("future return close in model fit")
    if as_of - end > timedelta(days=config["max_fit_age_days"]):
        return {
            **base_report,
            "status": "unavailable",
            "reason": "stale_fit_window",
            "fitEnd": end.isoformat(),
        }
    if any(set(panel[t]) != set(ids) for t in stamps):
        raise ValueError("complete common risk price grid required")
    if any(b - a != timedelta(days=1) for a, b in zip(stamps, stamps[1:])):
        raise ValueError("current crypto risk input requires daily UTC closes")
    rows = []
    for previous, stamp in zip(stamps, stamps[1:]):
        for identifier in ids:
            a, b = panel[previous][identifier], panel[stamp][identifier]
            rows.append(
                {
                    "symbol": str(identifier),
                    "timestamp": stamp,
                    "known_at": max(strict_json.clock(r["availableAt"]) for r in (a, b)),
                    "return": b["close"] / a["close"] - 1,
                }
            )
    fitted = estimate_market_risk(
        pl.DataFrame(rows),
        benchmark=str(config["benchmark_instrument_id"]),
        fit_end=end,
        model_known_at=as_of,
        currency=config["currency"],
        input_sha256=input_sha256,
        model_id="native-market-ols-v1",
    )
    built_at = datetime.now(UTC) if model_known_at is None else model_known_at
    if (
        not isinstance(built_at, datetime)
        or built_at.tzinfo is None
        or built_at.utcoffset() is None
        or built_at < as_of
        or built_at > datetime.now(UTC)
    ):
        raise ValueError("model production must follow input cutoff and precede verification")
    model = fitted["model"]
    model["model_known_at"] = built_at.isoformat()
    fitted["fit"]["inputCutoff"] = as_of.isoformat()
    fitted["fit"]["computedAt"] = built_at.isoformat()
    assets = {"INSTRUMENT:" + str(a["instrument_id"]): a for a in model["assets"]}
    assets["__CASH__"] = {"loadings": [0.0], "specific_variance": 0.0}
    weights = {}
    for name in ("qlib", "equal_weight_buy_hold"):
        w = native_snapshot_weights(runs[name], latest)
        weights[name] = w.filter(pl.col("timestamp") == end).sort("symbol")
        if weights[name].height != len(assets):
            raise ValueError("risk fit end and final native holdings snapshot must agree")
        weights[name] = weights[name].with_columns(
            pl.lit(built_at).alias("timestamp"), pl.lit(built_at).alias("known_at")
        )
    p, b = weights["qlib"], weights["equal_weight_buy_hold"]
    if p["symbol"].to_list() != b["symbol"].to_list():
        raise ValueError("native market risk holdings must align")
    exposure = p.select("timestamp", "symbol", "known_at").with_columns(
        pl.Series("market", [assets[s]["loadings"][0] for s in p["symbol"]])
    )
    specific = p.select("timestamp", "symbol", "known_at").with_columns(
        pl.Series("variance", [assets[s]["specific_variance"] for s in p["symbol"]])
    )
    covariance = pl.DataFrame({"factor": ["market"], "market": [model["covariance"][0][0]]})
    active = p.with_columns((p["weight"] - b["weight"]).alias("weight"))
    risk = {
        name: factor_risk_decomposition(
            w,
            exposure,
            covariance,
            specific,
            as_of=built_at,
            model_known_at=built_at,
            fit_end=end,
            model_id=model["model_id"],
            currency=model["currency"],
            horizon="one_day",
        )
        for name, w in {"portfolio": p, "benchmark": b, "active": active}.items()
    }
    return {
        **base_report,
        "status": "available",
        "asOf": built_at.isoformat(),
        "weightsSnapshotAt": end.isoformat(),
        "weightsBasis": "last_native_research_snapshot_declared_frozen_until_input_cutoff_not_live_positions",
        "cashAssumption": "explicit_zero_factor_and_specific_risk",
        "model": model,
        "fit": fitted["fit"],
        "modelSha256": hashlib.sha256(
            json.dumps(model, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest(),
        "risk": risk,
    }
