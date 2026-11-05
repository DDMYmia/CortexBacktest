"""Trusted adapter delivery preserves decimals and rejects unsafe inputs."""
import json
from datetime import UTC, datetime
from decimal import Decimal
from hashlib import sha256

import pytest

from cortexbacktest.pipelines import FactorValuation


def document():
    return {"accountRef": "binance-diagnostic", "sourceSha256": "a" * 64,
            "eventTime": "2026-10-08T00:00:00+00:00",
            "positions": [["INSTRUMENT:1", "0.125"]], "pendingSymbols": [],
            "valuation": {"sourceSha256": "a" * 64, "currency": "USDT", "equity": "410.0",
                          "eventTime": "2026-10-08T00:00:00+00:00",
                          "prices": [["INSTRUMENT:1", "100.25"]]}}


def load(content, **overrides):
    return FactorValuation.from_bytes(content, **{
        "expected_sha256": sha256(content).hexdigest(), "account_ref": "binance-diagnostic",
        "symbols": ("INSTRUMENT:1",), "now": datetime(2026, 10, 8, 0, 0, 1, tzinfo=UTC),
        "max_age_seconds": 60, **overrides})


def test_adapter_delivery_retains_exact_values():
    valuation = load(json.dumps(document()).encode())
    assert valuation.equity == Decimal("410.0")
    assert valuation.prices == (("INSTRUMENT:1", Decimal("100.25")),)
    assert valuation.portfolio.positions == (("INSTRUMENT:1", Decimal("0.125")),)


def test_explicit_empty_spot_capital_is_valid_only_with_zero_holdings():
    doc = document()
    doc["valuation"]["equity"] = "0"
    with pytest.raises(ValueError, match="Zero capital conflicts"):
        load(json.dumps(doc).encode())
    doc["positions"] = [["INSTRUMENT:1", "0"]]
    assert load(json.dumps(doc).encode()).equity == 0


@pytest.mark.parametrize("change", [
    lambda d: d.update(accountRef="different-account"),
    lambda d: d["valuation"].update(sourceSha256="b" * 64),
    lambda d: d["valuation"].update(equity=410.0),
    lambda d: d["valuation"].update(prices=[["INSTRUMENT:1", 100.25]]),
    lambda d: d["valuation"].update(prices=[]),
    lambda d: d.update(positions=[]),
    lambda d: d.update(pendingSymbols=["INSTRUMENT:1"]),
    lambda d: d.update(eventTime="2026-10-07T00:00:00+00:00"),
    lambda d: d["valuation"].update(equity="1.123456789012345678"),
])
def test_invalid_delivery_rejected(change):
    doc = document()
    change(doc)
    with pytest.raises(ValueError):
        load(json.dumps(doc).encode())


def test_changed_bytes_and_duplicate_fields_rejected():
    content = json.dumps(document()).encode()
    with pytest.raises(ValueError, match="digest"):
        load(content + b" ", expected_sha256=sha256(content).hexdigest())
    duplicate = content[:-1] + b', "accountRef": "binance-diagnostic"}'
    with pytest.raises(ValueError, match="Duplicate"):
        load(duplicate)
