"""External predictions need native source checks and explicit diagnostic scope."""
from datetime import datetime, timedelta, timezone
import hashlib
import json

import pytest

from cortexbacktest.adapters.qlib_predictions import evaluate_qlib_predictions, load_qlib_predictions


@pytest.fixture
def bundle(tmp_path):
    start = datetime(2020, 1, 1, tzinfo=timezone.utc)
    dates = [(start + timedelta(days=n)).date().isoformat() for n in range(120)]
    cutoff = "2021-01-01T00:00:00+00:00"
    records = []
    sessions = []
    for i in [1, 2, 3]:
        for n, day in enumerate(dates):
            close = (start + timedelta(days=n + 1)).isoformat()
            price = 100 + i * n / 10 + (n % 7)
            sessions.append({"instrumentId": str(i), "sessionDate": day, "sessionClose": close})
            records.append({"contractVersion": "market-bars-v4", "instrumentId": str(i), "sessionDate": day,
                "adjustment": "unadjusted", "ohlcBasis": "unadjusted", "source": "BINANCE",
                "adjustmentProvenance": "TEST-synthetic", "sessionClose": close,
                "sourceKnownAt": close, "recordedAt": close, "availableAt": close, "asOf": cutoff,
                "open": price, "high": price + 2, "low": price - 2, "close": price,
                "adjClose": price, "volume": 100000, "invalidReason": None})
    doc = {"contractVersion": "observation-file-v1", "recordVersion": "market-bars-v4", "asOf": cutoff,
        "scope": {"start": dates[0], "end": dates[-1], "source": "BINANCE", "adjustment": "unadjusted",
                  "sessions": sessions}, "recordCount": len(records), "records": records}
    raw = json.dumps(doc).encode()
    bars = tmp_path / "bars.json"
    bars.write_bytes(raw)
    predictions = [{"sessionDate": day, "instrumentId": str(i), "score": i * .1}
                   for day in dates[100:] for i in [1, 2, 3]]
    root = tmp_path / "worker"
    root.mkdir()
    (root / "model.txt").write_text("TEST-not-executed")
    (root / "predictions.json").write_text(json.dumps(predictions))
    manifest = {"contractVersion": "qlib-predictions-v1", "mode": "historical_reconstruction",
        "inputSha256": hashlib.sha256(raw).hexdigest(), "inputAsOf": cutoff,
        "settings": {"splits": {"train": [dates[60], dates[79]], "valid": [dates[82], dates[96]],
                                "test": [dates[100], dates[-1]]},
            "label": "Ref($close,-2)/Ref($close,-1)-1", "labelHorizon": 2,
            "features": "Alpha158-OHLCV", "excludedFeatures": ["VWAP0"], "rounds": 40,
            "earlyStopping": 10, "seed": 7, "trials": 1,
            "featureProcessors": ["ProcessInf", "ZScoreNorm-training-only", "Fillna"]},
        "instruments": ["1", "2", "3"], "recordCount": len(predictions),
        "generatedAt": "2021-01-02T00:00:00+00:00", "producerSha256": "a" * 64,
        "versions": {"pyqlib": "TEST", "lightgbm": "TEST"}, "featureCount": 157,
        "files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir()},
        "researchOnly": True, "qualificationIssued": False, "executionAuthorized": False}

    def save():
        data = json.dumps(manifest).encode()
        (root / "manifest.json").write_bytes(data)
        return {"directory": root, "manifest_sha256": hashlib.sha256(data).hexdigest(),
                "bars_path": bars, "bars_sha256": hashlib.sha256(raw).hexdigest(), "producer_sha256": "a" * 64}

    return root, manifest, save


def test_native_execution_keeps_three_portfolios_and_no_authority(bundle):
    _, _, save = bundle
    report, runs = evaluate_qlib_predictions(**save(), initial_cash=10000, commission_bps=10,
        slippage_bps=5, cash_rate_bps=100)
    assert set(runs) == {"qlib", "equal_weight_buy_hold", "cash"}
    assert runs["qlib"].records.fills().height > 0
    assert runs["cash"].records.fills().height == 0
    assert set(runs["equal_weight_buy_hold"].records.fills()["symbol"]) == {f"INSTRUMENT:{i}" for i in [1, 2, 3]}
    assert report["execution"]["cashBuffer"] > 0
    assert report["qualificationIssued"] is False and report["executionAuthorized"] is False


@pytest.mark.parametrize("mutation", ["authority", "embargo", "source", "method"])
def test_rehashed_untrusted_claims_are_refused(bundle, mutation):
    _, manifest, save = bundle
    if mutation == "authority":
        manifest["qualificationIssued"] = True
    elif mutation == "embargo":
        manifest["settings"]["splits"]["valid"][0] = manifest["settings"]["splits"]["train"][1]
    elif mutation == "source":
        manifest["inputSha256"] = "b" * 64
    else:
        manifest["settings"]["seed"] = 99
    with pytest.raises(ValueError):
        load_qlib_predictions(**save())


@pytest.mark.parametrize("kind", ["duplicate", "missing", "nan", "outside"])
def test_even_rehashed_bad_predictions_are_refused(bundle, kind):
    root, manifest, save = bundle
    rows = json.loads((root / "predictions.json").read_bytes())
    if kind == "duplicate":
        rows.append(rows[0])
    elif kind == "missing":
        rows.pop()
    elif kind == "nan":
        rows[0]["score"] = float("nan")
    else:
        rows[0]["sessionDate"] = "2019-01-01"
    raw = json.dumps(rows).encode()
    (root / "predictions.json").write_bytes(raw)
    manifest["files"]["predictions.json"] = hashlib.sha256(raw).hexdigest()
    manifest["recordCount"] = len(rows)
    with pytest.raises(ValueError):
        load_qlib_predictions(**save())


def test_changed_model_bytes_are_refused(bundle):
    root, _, save = bundle
    args = save()
    (root / "model.txt").write_text("changed")
    with pytest.raises(ValueError, match="digest"):
        load_qlib_predictions(**args)


@pytest.mark.parametrize('engine', ['fusion-native-v1', 'fusion-native-v2', 'fusion-native-v999'])
def test_native_alpha_engine_versions_are_explicit(bundle, engine):
    _, manifest, save = bundle
    manifest['versions'] = dict(alphaEngine=engine, numpy='TEST', pandas='TEST', lightgbm='TEST', pyarrow='TEST')
    if engine.endswith('v999'):
        with pytest.raises(ValueError, match='Invalid Qlib predictions manifest'):
            load_qlib_predictions(**save())
    else:
        loaded, _, _ = load_qlib_predictions(**save())
        assert loaded['versions']['alphaEngine'] == engine
