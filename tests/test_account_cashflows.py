import hashlib
import json
from pathlib import Path

import pytest

from cortexbacktest.adapters.account_cashflows import account_cashflow_performance


def document():
    return json.loads(
        (
            Path(__file__).parent
            / "fixtures/contracts/account-report-v1/valid-daily-returns.json"
        ).read_text()
    )


def save(tmp_path, doc):
    report = json.dumps(
        doc["report"], sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    doc["sourceReportSha256"] = hashlib.sha256(report).hexdigest()
    raw = json.dumps(doc).encode()
    path = tmp_path / "account.json"
    path.write_bytes(raw)
    return account_cashflow_performance(path, sha256=hashlib.sha256(raw).hexdigest())


def test_existing_native_fixture_does_not_claim_complete_cashflow_returns(tmp_path):
    result = save(tmp_path, document())
    assert result.summary["status"] == "insufficient_evidence"
    assert result.summary["twr"] is None and result.summary["mwr"] is None


def full_arithmetic_fixture():
    # Synthetic low-level arithmetic case, not FQ acceptance or broker authenticity.
    doc = document()
    report = doc["report"]
    report["aggregation"]["cash_transactions_covered"] = True
    report["missing_sections"] = []
    report["included_cash_event_ids"] = ["b" * 64]
    return doc


def test_explicit_fx_and_separate_broker_twr_day_basis(tmp_path):
    result = save(tmp_path, full_arithmetic_fixture())
    assert result.summary["twr"]["totalReturn"] == pytest.approx(0.0302)
    assert result.summary["netExternalFlow"] == 500.0
    assert result.summary["twrClockPolicy"].startswith("symbolic_day")
    assert result.summary["mwrClockPolicy"] == "broker_report_dates_day_resolution"
    assert result.to_dict()["executionAuthorized"] is False


def test_a_day_with_missing_evidence_withholds_returns(tmp_path):
    doc = full_arithmetic_fixture()
    doc["report"]["daily_returns"]["days"][1]["issues"] = ["missing_fx_rate"]
    assert save(tmp_path, doc).summary["reason"] == "incomplete_broker_return_evidence"


def zero_capital_fixture():
    doc = full_arithmetic_fixture()
    report = doc["report"]
    block = report["daily_returns"]
    report["from"] = "2025-12-31"
    report["opening_date"] = "2025-12-30"
    report["included_cash_event_ids"].append("a" * 64)
    first = block["days"][0]
    first.update(
        prior_nav="0",
        flow="1000",
        flow_sources=[{"event_id": "a" * 64, "amount": "1000", "currency": "USD", "rate": "1"}],
    )
    empty = {
        "date": "2025-12-31",
        "prior_date": "2025-12-30",
        "prior_nav": "0",
        "nav": "0",
        "twr": "0",
        "flow": "0",
        "issues": [],
    }
    block["days"].insert(0, empty)
    return doc


def test_source_backed_zero_capital_is_not_an_imputed_return(tmp_path):
    result = save(tmp_path, zero_capital_fixture())
    assert result.summary["status"] == "available"
    assert result.summary["twr"]["totalReturn"] == pytest.approx(0.0302)
    assert result.summary["twr"]["zeroCapitalPeriods"] == 1
    assert result.periods.height == 3 and result.periods["return"][0] is None
    assert result.summary["netExternalFlow"] == 1500.0
