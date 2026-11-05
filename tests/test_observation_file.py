from copy import deepcopy
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path

import pytest
from jsonschema.exceptions import ValidationError

from cortexbacktest.adapters.observation_file import (
    load_observation_file,
    validate_observation_file,
)

FIXTURES = Path(__file__).parent / "fixtures/contracts"


def document():
    return json.loads((FIXTURES / "observation-file-v1/valid-minimal.json").read_text())


def test_complete_file_preserves_receipts_and_checks_external_hash(tmp_path):
    payload = document()
    raw = json.dumps(payload).encode()
    path = tmp_path / "observations.json"
    path.write_bytes(raw)
    cutoff = datetime.fromisoformat(payload["asOf"])
    assert load_observation_file(path, sha256=sha256(raw).hexdigest(), as_of=cutoff) == payload
    path.write_bytes(raw + b" ")
    with pytest.raises(ValueError, match="digest"):
        load_observation_file(path, sha256=sha256(raw).hexdigest(), as_of=cutoff)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda p: p.update(recordCount=2),
        lambda p: p.update(recordVersion="future-v2"),
        lambda p: p.update(records=p["records"] * 2, recordCount=2),
        lambda p: p["records"][0].update(instrument_id=None),
        lambda p: p["records"][0].update(identity_known_at="2026-09-19T00:00:00Z"),
        lambda p: p["records"][0].update(known_at="2026-09-17T00:00:00Z"),
        lambda p: p["records"][0].update(value=float("nan")),
        lambda p: p["records"][0].update(metadata={"bad": float("inf")}),
        lambda p: p["records"][0].update(symbol="UNKNOWN"),
        lambda p: p["records"][0].update(symbol_source="another-provider"),
        lambda p: p["records"][0].update(as_of="2026-09-17T00:00:00Z"),
        lambda p: p["records"][0].pop("entity_receipt_id"),
        lambda p: p["records"][0].update(entity_receipt_id=None),
        lambda p: p["scope"].update(end="2026-09-16"),
    ],
)
def test_invalid_factors_are_rejected_instead_of_filtered(mutation):
    payload = document()
    cutoff = datetime.fromisoformat(payload["asOf"])
    mutation(payload)
    with pytest.raises((ValueError, ValidationError)):
        validate_observation_file(payload, as_of=cutoff)


def test_macro_tombstone_is_preserved_and_cannot_hide_future_receipt():
    row = json.loads((FIXTURES / "macro-observations-v2/valid-minimal.json").read_text())
    row.update(operation="delete", value=None, valueRaw=None)
    payload = {
        "contractVersion": "observation-file-v1",
        "recordVersion": "macro-observations-v2",
        "recordCount": 1,
        "asOf": row["asOf"],
        "scope": {
            "seriesId": row["seriesId"],
            "start": row["observationDate"],
            "end": row["observationDate"],
        },
        "records": [row],
    }
    cutoff = datetime.fromisoformat(row["asOf"])
    assert validate_observation_file(payload, as_of=cutoff)["records"][0]["operation"] == "delete"
    broken = deepcopy(payload)
    broken["records"][0]["recordedAt"] = "2099-01-01T00:00:00Z"
    with pytest.raises(ValueError, match="availability"):
        validate_observation_file(broken, as_of=cutoff)


def test_different_caller_cutoff_is_not_silently_accepted():
    with pytest.raises(ValueError, match="cutoff"):
        validate_observation_file(document(), as_of=datetime.fromisoformat("2025-01-01T00:00:00Z"))


def market_document():
    row = json.loads((FIXTURES / "market-bars-v2/valid-minimal.json").read_text())
    return {
        "contractVersion": "observation-file-v1",
        "recordVersion": "market-bars-v2",
        "recordCount": 1,
        "asOf": row["asOf"],
        "records": [row],
        "scope": {
            "source": row["source"],
            "adjustment": row["adjustment"],
            "start": row["sessionDate"],
            "end": row["sessionDate"],
            "sessions": [
                {key: row[key] for key in ("instrumentId", "sessionDate", "sessionClose")}
            ],
        },
    }


def test_market_file_binds_exact_sessions_and_raw_basis():
    payload = market_document()
    assert (
        validate_observation_file(payload, as_of=datetime.fromisoformat(payload["asOf"])) == payload
    )


def test_v4_invalidation_keeps_receipts_but_cannot_carry_usable_prices():
    payload = market_document()
    payload["recordVersion"] = "market-bars-v4"
    row = payload["records"][0]
    row.update(contractVersion="market-bars-v4", invalidReason="unverified_price_basis")
    for field in ("open", "high", "low", "close", "adjClose", "volume", "ohlcBasis", "adjustmentProvenance"):
        row[field] = None
    cutoff = datetime.fromisoformat(payload["asOf"])
    assert validate_observation_file(payload, as_of=cutoff) == payload
    row["close"] = 10
    with pytest.raises(ValidationError):
        validate_observation_file(payload, as_of=cutoff)
    row["close"] = None
    row["availableAt"] = "2020-01-01T00:00:00Z"
    with pytest.raises(ValueError, match="availability"):
        validate_observation_file(payload, as_of=cutoff)


def test_revision_file_keeps_receipt_identity_without_weakening_snapshot_uniqueness():
    payload = market_document()
    original = payload["records"][0]
    revised = {**original, "recordedAt": "2020-01-03T12:00:00Z", "availableAt": "2020-01-03T12:00:00Z"}
    payload["records"].append(revised)
    payload["recordCount"] = 2
    cutoff = datetime.fromisoformat(payload["asOf"])
    with pytest.raises(ValueError, match="Duplicate observation"):
        validate_observation_file(payload, as_of=cutoff)
    payload["recordVersion"] = "market-bars-v4"
    for row in payload["records"]:
        row.update(contractVersion="market-bars-v4", invalidReason=None)
    assert validate_observation_file(payload, as_of=cutoff) == payload
    revised.update(recordedAt=original["recordedAt"], availableAt=original["availableAt"])
    with pytest.raises(ValueError, match="Duplicate observation"):
        validate_observation_file(payload, as_of=cutoff)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda p: p.update(records=[], recordCount=0),
        lambda p: p["records"][0].update(instrumentId="2"),
        lambda p: p["records"][0].update(source="other"),
        lambda p: p["records"][0].update(sessionClose="2020-01-02T20:00:00Z"),
        lambda p: p["records"][0].update(sourceKnownAt="2020-01-05T00:00:00Z"),
        lambda p: p["records"][0].update(availableAt="2020-01-02T21:01:00Z"),
        lambda p: p["records"][0].update(high=9),
        lambda p: p["records"][0].update(adjClose=12),
        lambda p: p["scope"]["sessions"].append(
            {**p["scope"]["sessions"][0], "sessionClose": "2020-01-02T20:00:00Z"}
        ),
        lambda p: p["scope"]["sessions"].append({**p["scope"]["sessions"][0], "instrumentId": "2"}),
    ],
)
def test_invalid_market_scope_or_receipts_are_rejected(mutation):
    payload = market_document()
    mutation(payload)
    with pytest.raises((ValueError, ValidationError)):
        validate_observation_file(payload, as_of=datetime.fromisoformat(payload["asOf"]))


def test_filing_dated_factor_is_known_from_its_score_not_its_later_recording():
    payload = document()
    record = payload["records"][0]
    payload["scope"]["factorName"] = "institutional.share_change_qoq"
    record.update(
        source="fusionquant", factor_name="institutional.share_change_qoq",
        recorded_at="2026-09-20T00:00:00Z", metadata={"knowledge": "filing-date", "cik": "0000320193"},
    )
    record["known_at"] = record["identity_known_at"] = "2026-09-18T00:00:00Z"
    cutoff = datetime.fromisoformat(payload["asOf"])
    validate_observation_file(payload, as_of=cutoff)
    record["known_at"] = "2026-09-20T00:00:00Z"
    with pytest.raises(ValueError, match="availability"):
        validate_observation_file(payload, as_of=cutoff)
