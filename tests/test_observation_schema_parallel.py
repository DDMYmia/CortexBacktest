"""The chunked schema check accepts and rejects exactly what the whole-document check does."""
import copy

import pytest
from jsonschema import ValidationError

from cortexbacktest.adapters import observation_file as of


def _document(n: int) -> dict:
    sessions = [{"instrumentId": str(i % 7 + 1), "sessionDate": f"2024-01-{i % 28 + 1:02d}",
                 "sessionClose": "2024-01-02T21:00:00+00:00"} for i in range(n)]
    records = [{"instrumentId": "1", "sessionDate": "2024-01-02", "sessionClose": "2024-01-02T21:00:00+00:00",
                "availableAt": "2024-01-02T21:00:00+00:00", "recordedAt": "2024-01-02T21:00:00+00:00",
                "sourceKnownAt": "2024-01-02T21:00:00+00:00", "asOf": "2024-01-03T00:00:00+00:00",
                "contractVersion": "market-bars-v4", "source": "FMP", "adjustment": "total_return",
                "ohlcBasis": "unadjusted", "adjustmentProvenance": "fmp", "invalidReason": None,
                "open": 1., "high": 1., "low": 1., "close": 1., "volume": 1., "adjClose": 1.}] * n
    return {"contractVersion": "observation-file-v1", "recordVersion": "market-bars-v4", "recordCount": n,
            "asOf": "2024-01-03T00:00:00+00:00", "records": records,
            "scope": {"start": "2024-01-01", "end": "2024-01-31", "source": "FMP", "adjustment": "total_return",
                      "sessions": sessions}}


@pytest.fixture
def small_threshold(monkeypatch):
    monkeypatch.setattr(of, "_PARALLEL_ITEMS", 10)


def test_a_valid_large_file_passes_the_chunked_check(small_threshold):
    document = _document(400)
    of._VALIDATOR.validate(document)  # the whole-document check agrees
    of._validate_schema(document)


@pytest.mark.parametrize("where", ["records", "sessions"])
@pytest.mark.parametrize("position", [0, 199, 399])
def test_one_bad_item_anywhere_is_found(small_threshold, where, position):
    document = copy.deepcopy(_document(400))
    if where == "records":
        document["records"][position] = {**document["records"][position], "close": -1.}
    else:
        document["scope"]["sessions"][position] = {**document["scope"]["sessions"][position], "sessionDate": 5}
    with pytest.raises(ValidationError):
        of._VALIDATOR.validate(document)
    with pytest.raises(ValidationError):
        of._validate_schema(document)
