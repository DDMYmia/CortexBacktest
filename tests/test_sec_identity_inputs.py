"Small independent SEC contract examples, including a ticker shared with debt."

from copy import deepcopy
from datetime import datetime
import hashlib
import json

import pytest

from cortexbacktest.adapters.sec_identity import verify_sec_identities


def digest(value):
    return hashlib.sha256(value).hexdigest()


def fact(name, value, context="stock"):
    return f'<ix:nonNumeric contextRef="{context}" name="dei:{name}">{value}</ix:nonNumeric>'


@pytest.fixture
def received():
    cutoff = "2026-10-25T11:00:00+00:00"
    document = {
        "contractVersion": "received-sec-identities-v1",
        "asOf": cutoff,
        "historicalIdentityCertified": False,
        "qualificationIssued": False,
        "executionAuthorized": False,
        "records": [],
    }
    targets = {}
    for identifier in range(1, 11):
        symbol, cik = f"EX{identifier}", str(identifier).zfill(10)
        accession = "0001628280-26-000001"
        cover = "".join(
            fact(k, v)
            for k, v in [
                ("EntityCentralIndexKey", cik),
                ("EntityRegistrantName", "Example Inc."),
                ("DocumentType", "10-Q"),
                ("DocumentPeriodEndDate", "March 31, 2026"),
                ("TradingSymbol", symbol),
                ("Security12bTitle", "Common stock"),
                ("SecurityExchangeName", "Nasdaq"),
            ]
        )
        cover += "".join(
            fact(k, v, "bond")
            for k, v in [
                ("TradingSymbol", symbol),
                ("Security12bTitle", "3% Notes due 2030"),
                ("SecurityExchangeName", "New York Stock Exchange"),
            ]
        )
        data = {
            "cik": cik,
            "filings": {
                "recent": {
                    "accessionNumber": [accession],
                    "form": ["10-Q"],
                    "primaryDocument": ["example.htm"],
                    "filingDate": ["2026-05-01"],
                    "reportDate": ["2026-03-31"],
                    "acceptanceDateTime": ["2026-05-01T20:00:00Z"],
                }
            },
        }
        evidence = {"historical_identity_certified": False}
        for name, raw in [("submissions", json.dumps(data).encode()), ("filing", cover.encode())]:
            url = (
                f"https://data.sec.gov/submissions/CIK{cik}.json"
                if name == "submissions"
                else f"https://www.sec.gov/Archives/edgar/data/{identifier}/{accession.replace('-', '')}/example.htm"
            )
            receipt = {
                "contractVersion": "research-source-receipt-v1",
                "source": "sec",
                "dataset": name,
                "status": "received",
                "httpStatus": 200,
                "rawFile": "raw.data",
                "rawBytes": len(raw),
                "rawSha256": digest(raw),
                "receivedAt": "2026-10-25T10:00:00Z",
                "request": {"url": url},
                "scope": {
                    "cik": cik,
                    "accessionNumber": accession,
                    "primaryDocument": "example.htm",
                },
            }
            evidence[name + "_receipt_utf8"] = json.dumps(receipt)
            evidence[name + "_raw_utf8"] = raw.decode()
        pair = {
            name + "Sha256": digest(evidence[name + "_receipt_utf8"].encode())
            for name in ["submissions", "filing"]
        }
        attrs = {
            "contract_version": "sec-filing-identity-v1",
            "symbol": symbol,
            "cik": cik,
            "issuer_name": "Example Inc.",
            "security_title": "Common stock",
            "exchange_mic": "XNAS",
            "accession_number": accession,
            "form": "10-Q",
            "report_period_end": "2026-03-31",
            "filing_date": "2026-05-01",
            "source_accepted_at": "2026-05-01T20:00:00+00:00",
            "source_published_at": None,
            "source_time_policy": "SEC_acceptance_is_not_a_dissemination_timestamp",
            "received_at": "2026-10-25T10:00:00+00:00",
            "valid_from": None,
            "valid_to": None,
            "historical_identity_certified": False,
            "qualification_issued": False,
            "execution_authorized": False,
        }
        document["records"].append(
            {
                "instrumentId": identifier,
                "receivedAt": attrs["received_at"],
                "recordedAt": "2026-10-25T10:30:00+00:00",
                "sourceSha256": digest(
                    json.dumps(pair, sort_keys=True, separators=(",", ":")).encode()
                ),
                "attributes": attrs,
                "evidence": evidence,
            }
        )
        targets[identifier] = {"primary_symbol": symbol, "cik": cik, "exchange_mic": "NASDAQ"}
    return document, targets, datetime.fromisoformat(cutoff)


def verify(received):
    document, targets, cutoff = received
    return verify_sec_identities(
        document,
        targets,
        as_of=cutoff,
        venues={"NASDAQ": "XNAS", "Nasdaq": "XNAS", "New York Stock Exchange": "XNYS"},
    )


def test_same_ticker_debt_does_not_override_common_stock_and_acceptance_is_not_publication(
    received,
):
    clocks, summary = verify(received)
    assert len(clocks) == 10 and summary["snapshots"] == 10
    assert summary["knownAt"] == "2026-10-25T10:30:00+00:00"
    assert summary["sourcePublicationTimeAvailable"] is False
    assert summary["continuousHistoricalIdentityCertified"] is False


@pytest.mark.parametrize(
    "case", ["missing", "duplicate", "publication", "interval", "late_database", "naive_database"]
)
def test_incomplete_or_backdated_identity_claims_reject(received, case):
    document, _, _ = received
    row = document["records"][0]
    if case == "missing":
        document["records"].pop()
    elif case == "duplicate":
        document["records"][-1] = deepcopy(row)
    elif case == "publication":
        row["attributes"]["source_published_at"] = row["attributes"]["source_accepted_at"]
    elif case == "interval":
        row["attributes"]["valid_from"] = row["attributes"]["report_period_end"]
    elif case == "late_database":
        row["recordedAt"] = "2026-10-25T12:00:00+00:00"
    else:
        row["recordedAt"] = "2026-10-25T10:30:00"
    with pytest.raises(ValueError):
        verify(received)
