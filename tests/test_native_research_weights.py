from datetime import UTC, datetime
import hashlib
import json

import pytest

from cortexbacktest.adapters.native_research_weights import native_research_weights

IDS = list(range(1, 11))
STAMP = "2026-10-23T20:00:00+00:00"
CUTOFF = datetime(2026, 10, 25, tzinfo=UTC)


def files_for_weights(instruments=None):
    ids = IDS if instruments is None else instruments
    files = {}
    for role in ("portfolio", "benchmark"):
        metadata = (
            {"factor": "momentum:20:1"}
            if role == "portfolio"
            else {"benchmarkPolicy": "initial_equal_weight_buy_hold"}
        )
        manifest = {
            "symbols": [f"INSTRUMENT:{i}" for i in ids],
            "metadata": metadata,
            "data_sha256": "a" * 64,
            "start": "2026-07-01",
            "end": STAMP,
            "broker": {"initial_cash": 1000},
        }
        files[f"{role}-run_manifest.json"] = json.dumps(manifest).encode()
        # Portfolio owns 2 shares worth 200, versus benchmark's 9 shares worth 900; keep cash explicit.
        value = 200 if role == "portfolio" else 900
        files[f"{role}-positions.ndjson"] = json.dumps(
            {
                "symbol": f"INSTRUMENT:{ids[0]}",
                "quantity": value / 100,
                "last_price": 100,
                "market_value": value,
            }
        ).encode()
        files[f"{role}-equity_curve.ndjson"] = json.dumps(
            {"timestamp": STAMP, "equity": 1000, "cash": 1000 - value, "pending_cash": 0}
        ).encode()
    source = {
        "contractVersion": "native-research-weights-v1",
        "jobId": "b" * 32,
        "candidate": "momentum:20:1",
        "currency": "USD",
        "snapshotAt": STAMP,
        "knownAt": "2026-10-24T12:00:00+00:00",
        "instruments": ids,
        "sourceInstruments": ids,
        "accountPortfolio": False,
        "artifacts": {name: hashlib.sha256(raw).hexdigest() for name, raw in files.items()},
    }
    files["weight-source.json"] = json.dumps(source).encode()
    return files


def rehash(files):
    source = json.loads(files["weight-source.json"])
    source["artifacts"] = {
        name: hashlib.sha256(raw).hexdigest()
        for name, raw in files.items()
        if name != "weight-source.json"
    }
    files["weight-source.json"] = json.dumps(source).encode()


def test_native_weights_reconcile_cash_and_never_normalise_security_subset():
    result = native_research_weights(files_for_weights(), instruments=IDS, cutoff=CUTOFF)
    assert result["weights"]["portfolio"]["INSTRUMENT:1"] == 0.2
    assert result["weights"]["portfolio"]["__CASH__"] == 0.8
    assert result["weights"]["benchmark"]["INSTRUMENT:1"] == 0.9
    assert result["weights"]["benchmark"]["__CASH__"] == 0.1
    assert all(result["weights"]["portfolio"][f"INSTRUMENT:{i}"] == 0 for i in range(2, 11))


def dividend_files(instruments=None):
    ids = IDS if instruments is None else instruments
    files = files_for_weights(ids)
    for role in ("portfolio", "benchmark"):
        manifest = json.loads(files[f"{role}-run_manifest.json"])
        manifest['broker']['settlement_model'] = {'settings': {'delay_days': 0}}
        manifest['broker']['futures_symbols'] = []
        files[f"{role}-run_manifest.json"] = json.dumps(manifest).encode()
        files[f"{role}-corporate_action_events.ndjson"] = b''
    curve = json.loads(files['benchmark-equity_curve.ndjson'])
    curve.update(cash=50, pending_cash=50)
    files['benchmark-equity_curve.ndjson'] = json.dumps(curve).encode()
    event = {'symbol': f'INSTRUMENT:{ids[0]}', 'timestamp': '2026-10-22T13:30:00+00:00',
             'action_type': 'cash_dividend', 'cash_amount': 50, 'tax_amount': 0,
             'quantity_before': 9, 'quantity_after': 9, 'settles_on': '2026-11-01'}
    files['benchmark-corporate_action_events.ndjson'] = json.dumps(event).encode()
    rehash(files)
    return files


def test_unpaid_dividend_is_separate_from_spendable_cash_and_reconciles_equity():
    result = native_research_weights(dividend_files(), instruments=IDS, cutoff=CUTOFF)
    assert result['weights']['benchmark']['__CASH__'] == .05
    assert result['weights']['benchmark']['__DIVIDEND_RECEIVABLE__'] == .05
    assert result['weights']['benchmark']['INSTRUMENT:1'] == .9
    assert sum(result['weights']['benchmark'].values()) == pytest.approx(1)
    assert result['weights']['portfolio']['__DIVIDEND_RECEIVABLE__'] == 0
    assert result['balances']['benchmark']['dividendReceivable'] == 50


@pytest.mark.parametrize('case', ['missing', 'amount', 'paid', 'duplicate', 'future', 'unknown',
                                 'negative', 'boolean', 'trade', 'delayed_trade', 'futures'])
def test_rehashed_unevidenced_receivables_reject(case):
    files = dividend_files()
    name = 'benchmark-corporate_action_events.ndjson'
    event = json.loads(files[name])
    if case == 'missing':
        del files[name]
    elif case == 'duplicate':
        files[name] += b'\n' + files[name]
    elif case in {'delayed_trade', 'futures'}:
        for role in ('portfolio', 'benchmark'):
            manifest = json.loads(files[f'{role}-run_manifest.json'])
            if case == 'delayed_trade':
                manifest['broker']['settlement_model']['settings']['delay_days'] = 1
            else:
                manifest['broker']['futures_symbols'] = ['INSTRUMENT:1']
            files[f'{role}-run_manifest.json'] = json.dumps(manifest).encode()
    else:
        if case == 'amount':
            event['cash_amount'] = 49
        elif case == 'paid':
            event['settles_on'] = '2026-10-23'
        elif case == 'future':
            event['timestamp'] = '2026-10-26T13:30:00+00:00'
        elif case == 'unknown':
            event['symbol'] = 'INSTRUMENT:999'
        elif case == 'negative':
            event['cash_amount'] = -50
        elif case == 'boolean':
            event['cash_amount'] = True
        elif case == 'trade':
            event['action_type'] = 'trade_settlement'
        files[name] = json.dumps(event).encode()
    rehash(files)
    with pytest.raises(ValueError):
        native_research_weights(files, instruments=IDS, cutoff=CUTOFF)


@pytest.mark.parametrize(
    "case",
    [
        "valuation",
        "equity",
        "cash",
        "pending",
        "pending_bool",
        "short",
        "nonfinite",
        "duplicate",
        "unknown",
        "currency",
        "future",
        "naive",
        "snapshot",
        "policy",
        "scope",
        "partial",
        "tamper",
    ],
)
def test_fully_rehashed_inconsistent_native_holdings_reject(case):
    files = files_for_weights()
    position = json.loads(files["portfolio-positions.ndjson"])
    curve = json.loads(files["portfolio-equity_curve.ndjson"])
    source = json.loads(files["weight-source.json"])
    manifest = json.loads(files["benchmark-run_manifest.json"])
    if case == "valuation":
        position["market_value"] = 201
    elif case == "equity":
        curve["equity"] = 999
    elif case == "cash":
        curve["cash"] = -1
    elif case == "pending":
        curve["pending_cash"] = 1
    elif case == "pending_bool":
        curve["pending_cash"] = False
    elif case == "short":
        position["quantity"] = -2
    elif case == "nonfinite":
        position["last_price"] = float("nan")
    elif case == "unknown":
        position["symbol"] = "INSTRUMENT:999"
    elif case == "currency":
        source["currency"] = "EUR"
    elif case == "future":
        source["knownAt"] = "2099-01-01T00:00:00+00:00"
    elif case == "naive":
        source["snapshotAt"] = "2026-10-23T20:00:00"
    elif case == "snapshot":
        source["snapshotAt"] = "2026-10-22T20:00:00+00:00"
    elif case == "policy":
        manifest["metadata"]["benchmarkPolicy"] = "current_cap_weights"
    elif case == "scope":
        manifest["data_sha256"] = "c" * 64
    files["portfolio-positions.ndjson"] = json.dumps(position).encode()
    files["portfolio-equity_curve.ndjson"] = json.dumps(curve).encode()
    files["weight-source.json"] = json.dumps(source).encode()
    files["benchmark-run_manifest.json"] = json.dumps(manifest).encode()
    if case == "duplicate":
        files["portfolio-positions.ndjson"] += b"\n" + files["portfolio-positions.ndjson"]
    if case == "partial":
        files.pop("benchmark-positions.ndjson")
    if case != "tamper":
        rehash(files)
    else:
        files["portfolio-positions.ndjson"] = b"changed"
    with pytest.raises((ValueError, KeyError)):
        native_research_weights(files, instruments=IDS, cutoff=CUTOFF)


def test_risk_pool_can_cover_a_smaller_research_portfolio_without_dropping_cash():
    files = files_for_weights()
    source = json.loads(files["weight-source.json"])
    source["instruments"] = IDS + [11]
    files["weight-source.json"] = json.dumps(source).encode()
    result = native_research_weights(files, instruments=IDS + [11], cutoff=CUTOFF)
    assert result["weights"]["portfolio"]["INSTRUMENT:11"] == 0
    assert result["weights"]["portfolio"]["INSTRUMENT:1"] == 0.2
    assert result["weights"]["portfolio"]["__CASH__"] == 0.8
