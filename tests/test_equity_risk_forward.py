"Independent frozen-forecast math and future-only price/clock acceptance."

from copy import deepcopy
from datetime import UTC, datetime, timedelta
from statistics import NormalDist

import numpy as np
import polars as pl
import pytest
from jsonschema import ValidationError

from cortexbacktest.reporting.equity_risk_forward import (
    evaluate_equity_forecast,
    freeze_equity_forecast,
)

FROZEN = datetime(2025, 1, 1, 12, tzinfo=UTC)


def forecast_fixture():
    days = [FROZEN.date() + timedelta(days=i) for i in range(1, 65)]
    dates = [
        datetime.combine(d, datetime.min.time(), UTC).replace(hour=21)
        for d in days
        if d.weekday() < 5
    ][:31]
    sessions = {
        "calendar": "us_equity",
        "generatedAt": FROZEN.isoformat(),
        "closes": [{"sessionDate": str(d.date()), "sessionClose": d.isoformat()} for d in dates],
    }
    t = range(60)
    rows = [
        {
            "timestamp": FROZEN - timedelta(days=61 - j),
            "known_at": FROZEN,
            "symbol": str(i),
            "return": float(np.sin(j / 5 + i) * 0.02),
        }
        for j in t
        for i in range(1, 11)
    ]
    weights = pl.DataFrame(
        [
            {"symbol": f"INSTRUMENT:{i}", "weight": 0.1, "benchmark_weight": 0.2 if i <= 5 else 0.0}
            for i in range(1, 11)
        ]
    )
    report = {
        "asOf": FROZEN.isoformat(),
        "inputSha256": "a" * 64,
        "modelSha256": "b" * 64,
        "model": {"assets": [{"instrument_id": i} for i in range(1, 11)]},
        "risk": {
            role: {"totalVariance": var}
            for role, var in [("portfolio", 0.0004), ("benchmark", 0.0009), ("active", 0.0001)]
        },
    }
    returns = pl.DataFrame(rows)
    forecast = freeze_equity_forecast(
        report,
        returns,
        weights,
        {"sessions": 30, "confidence": 0.95, "significance": 0.05},
        sessions,
    )
    return forecast, dates, returns


def prices_fixture(forecast, dates):
    cutoff = dates[-1] + timedelta(hours=1)
    rows = []
    # Exactly five -5% days and twenty-five +1% days in the future window.
    for instrument in range(1, 11):
        value = 100.0
        for j, stamp in enumerate(dates):
            if j:
                value *= 0.95 if j % 6 == 0 else 1.01
            rows.append(
                {
                    "contractVersion": "market-bars-v4",
                    "instrumentId": str(instrument),
                    "sessionDate": str(stamp.date()),
                    "sessionClose": stamp.isoformat(),
                    "recordedAt": cutoff.isoformat(),
                    "sourceKnownAt": stamp.isoformat(),
                    "availableAt": cutoff.isoformat(),
                    "asOf": cutoff.isoformat(),
                    "source": "FMP",
                    "adjustment": "total_return",
                    "ohlcBasis": "unadjusted",
                    "adjustmentProvenance": "fixture source basis",
                    "invalidReason": None,
                    "open": value,
                    "high": value,
                    "low": value,
                    "close": value,
                    "volume": 10000.0,
                    "adjClose": value,
                }
            )
    doc = {
        "contractVersion": "observation-file-v1",
        "recordVersion": "market-bars-v4",
        "asOf": cutoff.isoformat(),
        "recordCount": len(rows),
        "records": rows,
        "scope": {
            "start": str(dates[0].date()),
            "end": str(dates[-1].date()),
            "source": "FMP",
            "adjustment": "total_return",
            "sessions": [
                {k: r[k] for k in ("instrumentId", "sessionDate", "sessionClose")} for r in rows
            ],
        },
    }
    return doc, cutoff


def test_frozen_gaussian_var_and_empirical_comparator_use_only_training_data():
    forecast, _, returns = forecast_fixture()
    assert forecast["portfolios"]["portfolio"]["modelVar"] == pytest.approx(
        NormalDist().inv_cdf(0.95) * 0.02
    )
    independent = [
        sum(r["return"] for r in returns.filter(pl.col("timestamp") == stamp).to_dicts()) / 10
        for stamp in sorted(returns["timestamp"].unique())
    ]
    assert forecast["portfolios"]["portfolio"]["empiricalVar"] == pytest.approx(
        max(0.0, -np.quantile(independent, 0.05))
    )
    assert all(datetime.fromisoformat(p["periodStart"]) > FROZEN for p in forecast["periods"])
    assert forecast["portfolios"]["portfolio"]["weights"]["__CASH__"] == 0


def test_waiting_does_not_invent_returns_or_calibration_statistics():
    forecast, _, _ = forecast_fixture()
    report = evaluate_equity_forecast(forecast, None, as_of=FROZEN.isoformat())
    assert report["status"] == "waiting_for_sessions" and report["observedIntervals"] == 0
    coverage = report["observationCoverage"]
    assert coverage["dueAtCutoff"] == coverage["requestedDue"] == coverage["acceptedIntervals"] == 0
    assert coverage["notYetDue"] == 30
    assert coverage["excludedByThrough"] == coverage["missingPriceIntervals"] == 0
    assert all(r["summary"] is None and r["breaches"] == 0 for r in report["portfolios"].values())
    with pytest.raises(ValueError, match="no future price input"):
        evaluate_equity_forecast(forecast, {}, as_of=FROZEN.isoformat())


def test_frozen_receivable_is_preserved_without_inventing_a_market_price_series():
    original, dates, returns = forecast_fixture()
    report = {'asOf': original['frozenAt'], 'inputSha256': original['inputSha256'],
              'modelSha256': original['modelSha256'],
              'model': {'assets': [{'instrument_id': i} for i in range(1, 11)]},
              'risk': {role: {'totalVariance': .0004} for role in ('portfolio', 'benchmark', 'active')}}
    rows = [{'symbol': f'INSTRUMENT:{i}', 'weight': .09, 'benchmark_weight': .1} for i in range(1, 11)]
    rows += [{'symbol': '__CASH__', 'weight': .05, 'benchmark_weight': 0.},
             {'symbol': '__DIVIDEND_RECEIVABLE__', 'weight': .05, 'benchmark_weight': 0.}]
    sessions = {'calendar': 'us_equity', 'generatedAt': original['frozenAt'],
                'closes': [{'sessionDate': str(d.date()), 'sessionClose': d.isoformat()} for d in dates]}
    frozen = freeze_equity_forecast(report, returns, pl.DataFrame(rows), original['config'], sessions)
    assert frozen['portfolios']['portfolio']['weights']['__DIVIDEND_RECEIVABLE__'] == .05
    assert frozen['portfolios']['active']['weights']['__DIVIDEND_RECEIVABLE__'] == .05
    assert frozen['returnPolicy'].endswith('_dividend_receivable_zero')
    prices, cutoff = prices_fixture(frozen, dates)
    observed = evaluate_equity_forecast(frozen, prices, as_of=cutoff.isoformat())
    assert observed['observedIntervals'] == 30
    assert observed['portfolios']['portfolio_model']['breaches'] == 5


def test_future_received_returns_match_oracle_and_calibration_counts():
    forecast, dates, _ = forecast_fixture()
    prices, cutoff = prices_fixture(forecast, dates)
    before = deepcopy(forecast)
    result = evaluate_equity_forecast(forecast, prices, as_of=cutoff.isoformat())
    assert result["status"] == "complete" and result["observedIntervals"] == 30
    assert result["portfolios"]["portfolio_model"]["summary"]["violations"] == 5
    assert result["portfolios"]["portfolio_model"]["summary"]["unconditionalCoverage"][
        "expectedViolations"
    ] == pytest.approx(1.5)
    assert result["portfolios"]["active_model"]["summary"]["violations"] == 0
    assert forecast == before and not result["qualificationIssued"]


@pytest.mark.parametrize(
    "case",
    ["missing", "invalid_latest", "future_receipt", "wrong_adjustment", "extra", "early_cutoff"],
)
def test_future_price_evidence_is_complete_and_cutoff_bound(case):
    forecast, dates, _ = forecast_fixture()
    prices, cutoff = prices_fixture(forecast, dates)
    if case == "missing":
        prices["records"].pop()
        prices["recordCount"] -= 1
    elif case == "invalid_latest":
        prices["records"][0]["invalidReason"] = "provider_basis_unknown"
    elif case == "future_receipt":
        prices["records"][0]["recordedAt"] = (cutoff + timedelta(seconds=1)).isoformat()
    elif case == "wrong_adjustment":
        prices["scope"]["adjustment"] = "unadjusted"
    elif case == "extra":
        forecast["portfolios"]["portfolio"]["weights"].pop("INSTRUMENT:10")
    elif case == "early_cutoff":
        cutoff = FROZEN - timedelta(seconds=1)
    with pytest.raises((ValueError, ValidationError)):
        evaluate_equity_forecast(forecast, prices, as_of=cutoff.isoformat())


def test_partial_observation_excludes_unrequested_future_end_and_keeps_statistics_unavailable():
    forecast, dates, _ = forecast_fixture()
    prices, cutoff = prices_fixture(forecast, dates)
    through = forecast["periods"][28]["sessionDate"]
    prices["records"] = [r for r in prices["records"] if r["sessionDate"] <= through]
    prices["recordCount"] = len(prices["records"])
    prices["scope"]["end"] = through
    prices["scope"]["sessions"] = [
        r for r in prices["scope"]["sessions"] if r["sessionDate"] <= through
    ]
    result = evaluate_equity_forecast(forecast, prices, as_of=cutoff.isoformat(), through=through)
    assert result["status"] == "partial" and result["observedIntervals"] == 29
    coverage = result["observationCoverage"]
    assert coverage["dueAtCutoff"] == 30 and coverage["requestedDue"] == coverage["acceptedIntervals"] == 29
    assert coverage["excludedByThrough"] == 1 and coverage["notYetDue"] == 0
    assert coverage["missingPriceIntervals"] == 0
    assert result["portfolios"]["portfolio_model"]["breaches"] == 4
    assert all(value["summary"] is None for value in result["portfolios"].values())


def test_latest_invalid_revision_is_not_replaced_by_an_older_valid_price():
    forecast, dates, _ = forecast_fixture()
    prices, cutoff = prices_fixture(forecast, dates)
    for row in prices["records"]:
        row["recordedAt"] = row["sourceKnownAt"]
        row["availableAt"] = row["sourceKnownAt"]
    latest = dict(prices["records"][0])
    latest.update(
        recordedAt=cutoff.isoformat(),
        availableAt=cutoff.isoformat(),
        invalidReason="unverified_price_basis",
        ohlcBasis=None,
        adjustmentProvenance=None,
    )
    for field in ("open", "high", "low", "close", "volume", "adjClose"):
        latest[field] = None
    prices["records"].append(latest)
    prices["recordCount"] += 1
    with pytest.raises(ValueError, match="latest prospective price revision invalid"):
        evaluate_equity_forecast(forecast, prices, as_of=cutoff.isoformat())
