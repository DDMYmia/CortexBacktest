from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math

import pytest

from test_native_group_attribution import fixture
from cortexbacktest.reporting.native_market_risk import native_market_risk, validate_quote_evidence

ASOF = datetime(2025, 1, 3, 2, tzinfo=timezone.utc)
CONFIG = {"benchmark_instrument_id": 1, "window": 60, "max_fit_age_days": 7, "currency": "USDT"}


def quotes():
    symbols = [
        {
            "symbol": base + "USDT",
            "baseAsset": base,
            "quoteAsset": "USDT",
            "status": "TRADING",
            "isSpotTradingAllowed": True,
        }
        for base in ["A", "B"]
    ]
    raw = json.dumps({"symbols": symbols})
    received = (ASOF - timedelta(hours=1)).isoformat()
    receipt = json.dumps(
        {
            "contractVersion": "research-source-receipt-v1",
            "rawFile": "raw.data",
            "rawBytes": len(raw.encode()),
            "rawSha256": hashlib.sha256(raw.encode()).hexdigest(),
            "source": "binance",
            "dataset": "exchange_info",
            "status": "received",
            "httpStatus": 200,
            "receivedAt": received,
        }
    )
    return {
        "contractVersion": "market-quote-evidence-v1",
        "asOf": ASOF.isoformat(),
        "currency": "USDT",
        "records": [
            {
                "instrumentId": i + 1,
                "symbol": x["symbol"],
                "receivedAt": received,
                "recordedAt": received,
                "receiptSha256": hashlib.sha256(receipt.encode()).hexdigest(),
                "receiptUtf8": receipt,
                "rawUtf8": raw,
            }
            for i, x in enumerate(symbols)
        ],
    }


def inputs():
    runs, latest, _ = fixture()
    for row in latest.values():
        row["availableAt"] = row["sessionClose"]
    first = datetime(2025, 1, 1, tzinfo=timezone.utc)
    for i in range(88):
        stamp = first - timedelta(days=88 - i)
        for instrument in ["1", "2"]:
            latest[instrument, str(i) + "history"] = {
                "instrumentId": instrument,
                "sessionClose": stamp.isoformat(),
                "availableAt": stamp.isoformat(),
                "close": 100 + math.sin(i / 4 + int(instrument)) * 3,
            }
    return runs, latest


def report(config=None, evidence=None, asof=ASOF):
    runs, latest = inputs()
    return native_market_risk(
        runs,
        latest,
        config or CONFIG,
        as_of=asof,
        input_sha256="a" * 64,
        quotes=evidence or quotes(),
        model_known_at=asof,
    )


def test_real_closing_holdings_current_cutoff_not_backdated_and_euler_reconciles():
    result = report()
    assert result["status"] == "available"
    assert result["weightsSnapshotAt"] == "2025-01-03T00:00:00+00:00"
    assert result["model"]["model_known_at"] == ASOF.isoformat()
    assert result["fit"]["observations"] == 60
    assert result["risk"]["portfolio"]["riskShareStatus"] == "undefined_zero_risk"
    assert result["risk"]["benchmark"]["totalVariance"] > 0
    for r in result["risk"].values():
        assert abs(r["factorVariance"] + r["specificVariance"] - r["totalVariance"]) < 1e-12
        assert abs(r["reconciliationResidual"]) < 1e-12
    assert "not_historical_forecast" in result["interpretation"]
    assert not result["qualificationIssued"] and not result["executionAuthorized"]


def test_insufficient_and_stale_inputs_do_not_emit_usable_model():
    assert report({**CONFIG, "window": 252})["reason"] == "insufficient_complete_history"
    evidence = quotes()
    future = ASOF + timedelta(days=8)
    evidence["asOf"] = future.isoformat()
    result = report(evidence=evidence, asof=future)
    assert result["reason"] == "stale_fit_window" and "model" not in result


@pytest.mark.parametrize("kind", ["digest", "clock", "scope", "currency", "duplicate", "symbol"])
def test_quote_evidence_missing_changed_future_or_wrong_units_reject(kind):
    evidence = deepcopy(quotes())
    if kind == "digest":
        evidence["records"][0]["rawUtf8"] += " "
    elif kind == "clock":
        evidence["records"][0]["recordedAt"] = (ASOF + timedelta(hours=1)).isoformat()
    elif kind == "scope":
        evidence["records"] = evidence["records"][:1]
    elif kind == "currency":
        evidence["currency"] = "USD"
    elif kind == "duplicate":
        evidence["records"] = [evidence["records"][0]] * 2
    else:
        evidence["records"][0]["symbol"] = "OTHER"
    with pytest.raises(ValueError):
        validate_quote_evidence(evidence, instruments=[1, 2], currency="USDT", as_of=ASOF)


def test_rehashing_foreign_quote_source_does_not_make_it_common_currency():
    evidence = quotes()
    for row in evidence["records"]:
        raw = json.loads(row["rawUtf8"])
        raw["symbols"][0]["quoteAsset"] = "USD"
        row["rawUtf8"] = json.dumps(raw)
        receipt = json.loads(row["receiptUtf8"])
        receipt["rawBytes"] = len(row["rawUtf8"].encode())
        receipt["rawSha256"] = hashlib.sha256(row["rawUtf8"].encode()).hexdigest()
        row["receiptUtf8"] = json.dumps(receipt)
        row["receiptSha256"] = hashlib.sha256(row["receiptUtf8"].encode()).hexdigest()
    with pytest.raises(ValueError, match="common spot quote"):
        report(evidence=evidence)
