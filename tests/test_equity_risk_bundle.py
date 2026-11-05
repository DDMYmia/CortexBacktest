"Received real-bundle tests run only with an explicit external input descriptor."

from datetime import datetime, timedelta
import hashlib
import json
import os
from pathlib import Path
import shutil
import re

import pytest

from cortexbacktest.reporting.native_equity_risk import fit_equity_risk_bundle


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def bundle(tmp_path):
    supplied = os.getenv("FUSION_EQUITY_RISK_DESCRIPTOR")
    if not supplied:
        pytest.skip("Explicit received equity bundle descriptor required")
    descriptor = json.loads(Path(supplied).read_text())
    path = Path(descriptor["path"])
    assert sha(path) == descriptor["sha256"]
    target = tmp_path / "bundle"
    shutil.copytree(path.parent, target)
    return target / "manifest.json"


def rehash(path):
    root = json.loads(path.read_text())
    style_path = path.with_name("style-manifest.json")
    style = json.loads(style_path.read_text())
    bindings_path = path.with_name("profile-bindings.json")
    bindings = json.loads(bindings_path.read_text())
    for i, binding in enumerate(bindings):
        binding["sha256"] = sha(path.with_name(f"profile-{i}.json"))
    bindings_path.write_text(json.dumps(bindings))
    identity_sha = sha(path.with_name("filing-identities.json"))
    style["identityInputSha256"] = identity_sha
    root["identityInputSha256"] = identity_sha
    style["files"] = {name: sha(path.with_name(name)) for name in style["files"]}
    style_path.write_text(json.dumps(style))
    root["styleInputSha256"] = sha(style_path)
    root["priceInputSha256"] = sha(path.with_name("equity-prices.json"))
    root["files"] = {name: sha(path.with_name(name)) for name in root["files"]}
    path.write_text(json.dumps(root))
    return sha(path)


def test_received_usd_model_reproduces_and_does_not_admit_historical_orders(bundle):
    first = fit_equity_risk_bundle(
        bundle, manifest_sha256=sha(bundle),
        model_known_at=datetime.fromisoformat(json.loads(bundle.read_bytes())["asOf"]),
    )
    again = fit_equity_risk_bundle(
        bundle,
        manifest_sha256=sha(bundle),
        model_known_at=datetime.fromisoformat(first["model"]["model_known_at"]),
    )
    assert first == again
    assert len(first["model"]["assets"]) == 10 and first["model"]["currency"] == "USD"
    assert first["model"]["factors"] == ["market", "size", "value"]
    assert first["fit"]["observations"] == 90
    assert first["identityEvidence"]["snapshots"] == 10
    assert first["identityEvidence"]["sourcePublicationTimeAvailable"] is False
    assert first["identityEvidence"]["continuousHistoricalIdentityCertified"] is False
    assert not first["executionAdmitted"] and not first["qualificationIssued"]
    assert all(abs(r["reconciliationResidual"]) < 1e-12 for r in first["risk"].values())


@pytest.mark.parametrize(
    "case", ["characteristic", "missing", "currency", "future_price", "latest_invalid", "authority"]
)
def test_rehashed_inconsistent_derived_data_currency_clocks_or_rights_reject(bundle, case):
    if case in {"characteristic", "missing"}:
        p = bundle.with_name("characteristics.json")
        rows = json.loads(p.read_text())
        if case == "characteristic":
            rows[0]["book_to_market"] += 0.1
        else:
            rows.pop()
        p.write_text(json.dumps(rows))
    elif case == "currency":
        p = bundle.with_name("profile-0.data")
        rows = json.loads(p.read_text())
        rows[0]["currency"] = "HKD"
        p.write_text(json.dumps(rows))
        r = bundle.with_name("profile-0.json")
        receipt = json.loads(r.read_text())
        receipt["rawSha256"] = sha(p)
        receipt["rawBytes"] = len(p.read_bytes())
        r.write_text(json.dumps(receipt))
    elif case in {"future_price", "latest_invalid"}:
        p = bundle.with_name("equity-prices.json")
        document = json.loads(p.read_text())
        row = dict(document["records"][0])
        cutoff = datetime.fromisoformat(document["asOf"])
        received = (
            cutoff + timedelta(seconds=1)
            if case == "future_price"
            else cutoff - timedelta(microseconds=1)
        )
        row["recordedAt"] = received.isoformat()
        row["availableAt"] = received.isoformat()
        if case == "latest_invalid":
            row.update(
                invalidReason="unverified_price_basis", ohlcBasis=None, adjustmentProvenance=None
            )
            for name in ["open", "high", "low", "close", "volume", "adjClose"]:
                row[name] = None
        document["records"].append(row)
        document["recordCount"] += 1
        p.write_text(json.dumps(document))
    else:
        p = bundle.with_name("execution-admission.json")
        admission = json.loads(p.read_text())
        admission["executionAdmitted"] = True
        p.write_text(json.dumps(admission))
    updated = rehash(bundle)
    with pytest.raises(ValueError):
        fit_equity_risk_bundle(bundle, manifest_sha256=updated)


@pytest.mark.parametrize(
    "case",
    [
        "missing",
        "cik",
        "venue",
        "bond",
        "context",
        "future_record",
        "publication",
        "validity",
        "pair",
        "known_at",
        "legacy",
    ],
)
def test_sec_identity_tampering_rejects_after_rebinding_every_digest(bundle, case):
    root = json.loads(bundle.read_text())
    path = bundle.with_name("filing-identities.json")
    document = json.loads(path.read_text())
    row = document["records"][0]
    body_changed = False
    if case == "missing":
        document["records"].pop()
    elif case == "cik":
        row["evidence"]["filing_raw_utf8"] = row["evidence"]["filing_raw_utf8"].replace(
            row["attributes"]["cik"], "9999999999"
        )
        row["attributes"]["cik"] = "9999999999"
        body_changed = True
    elif case == "venue":
        row["evidence"]["filing_raw_utf8"] = row["evidence"]["filing_raw_utf8"].replace(
            "The Nasdaq Stock Market LLC", "New York Stock Exchange"
        )
        row["attributes"]["exchange_mic"] = "XNYS"
        body_changed = True
    elif case in {"bond", "context"}:
        raw = row["evidence"]["filing_raw_utf8"]
        if case == "bond":
            raw, changes = re.subn(
                r'(<ix:nonNumeric\b[^>]*name="dei:Security12bTitle"[^>]*>).*?(</ix:nonNumeric>)',
                r"\g<1>Senior notes\g<2>",
                raw,
                flags=re.I | re.S,
            )
            row["attributes"]["security_title"] = "Senior notes"
        else:
            raw, changes = re.subn(
                r'(<ix:nonNumeric\b[^>]*name="dei:SecurityExchangeName"[^>]*contextRef=")[^"]+',
                r"\g<1>wrong-context",
                raw,
                flags=re.I,
            )
            if not changes:
                raw, changes = re.subn(
                    r'(<ix:nonNumeric\b[^>]*contextRef=")[^"]+("[^>]*name="dei:SecurityExchangeName")',
                    r"\g<1>wrong-context\g<2>",
                    raw,
                    flags=re.I,
                )
        assert changes
        row["evidence"]["filing_raw_utf8"] = raw
        body_changed = True
    elif case == "future_record":
        row["recordedAt"] = (
            datetime.fromisoformat(document["asOf"]) + timedelta(seconds=1)
        ).isoformat()
    elif case == "publication":
        row["attributes"]["source_published_at"] = row["attributes"]["source_accepted_at"]
    elif case == "validity":
        row["attributes"]["valid_from"] = row["attributes"]["report_period_end"]
    elif case == "pair":
        row["sourceSha256"] = "0" * 64
    elif case == "known_at":
        p = bundle.with_name("characteristics.json")
        records = json.loads(p.read_text())
        records[0]["known_at"] = row["attributes"]["source_accepted_at"]
        p.write_text(json.dumps(records))
    else:
        root["contractVersion"] = "equity-risk-inputs-v1"
        bundle.write_text(json.dumps(root))
    if body_changed:
        evidence = row["evidence"]
        receipt = json.loads(evidence["filing_receipt_utf8"])
        raw = evidence["filing_raw_utf8"].encode()
        receipt.update(rawSha256=hashlib.sha256(raw).hexdigest(), rawBytes=len(raw))
        evidence["filing_receipt_utf8"] = json.dumps(receipt)
        pair = {
            name + "Sha256": hashlib.sha256(evidence[name + "_receipt_utf8"].encode()).hexdigest()
            for name in ("submissions", "filing")
        }
        row["sourceSha256"] = hashlib.sha256(
            json.dumps(pair, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError):
        fit_equity_risk_bundle(bundle, manifest_sha256=rehash(bundle))


def test_recomputation_keeps_frozen_inputs_and_real_production_clock(bundle):
    manifest = json.loads(bundle.read_bytes())
    cutoff = datetime.fromisoformat(manifest["asOf"])
    first = fit_equity_risk_bundle(bundle, manifest_sha256=sha(bundle), model_known_at=cutoff)
    replay = fit_equity_risk_bundle(bundle, manifest_sha256=sha(bundle), retrospective=True)
    assert replay["analysisMode"] == "retrospective_recomputation"
    assert replay["inputAsOf"] == first["inputAsOf"]
    assert replay["inputSha256"] == first["inputSha256"]
    assert replay["model"]["assets"] == first["model"]["assets"]
    assert {k: v for k, v in replay["model"].items() if k != "model_known_at"} == {
        k: v for k, v in first["model"].items() if k != "model_known_at"
    }
    assert datetime.fromisoformat(replay["asOf"]) >= cutoff
    assert replay["historicalInputSemantics"]["certified"] is False
    assert not replay["executionAuthorized"]


@pytest.mark.parametrize('unpaid_dividends', [False, True])
def test_bound_native_holdings_replace_validation_weights_with_cash_and_oracle(bundle, unpaid_dividends):
    import numpy as np
    from test_native_research_weights import dividend_files, files_for_weights

    manifest = json.loads(bundle.read_bytes())
    files = (dividend_files if unpaid_dividends else files_for_weights)(manifest["instruments"])
    for name, raw in files.items():
        bundle.with_name(name).write_bytes(raw)
        manifest["files"][name] = hashlib.sha256(raw).hexdigest()
    bundle.write_text(json.dumps(manifest))
    report = fit_equity_risk_bundle(bundle, manifest_sha256=sha(bundle), retrospective=True)
    assert report["portfolioSource"]["weights"]["portfolio"]["__CASH__"] == .8
    if unpaid_dividends:
        assert report['portfolioSource']['weights']['benchmark']['__DIVIDEND_RECEIVABLE__'] == .05
        assert report['portfolioSource']['weights']['benchmark']['__CASH__'] == .05
        assert report['portfolioSource']['balances']['benchmark']['dividendReceivable'] == 50
    first_id = manifest["instruments"][0]
    asset = next(a for a in report["model"]["assets"] if a["instrument_id"] == first_id)
    loading = np.array(asset["loadings"])
    variance = float(loading @ np.array(report["model"]["covariance"]) @ loading + asset["specific_variance"])
    for role, weight in [("portfolio", .2), ("benchmark", .9), ("active", -.7)]:
        assert report["risk"][role]["totalVariance"] == pytest.approx(weight**2 * variance, abs=1e-18)
    assert report["portfolioDefinition"] == "owned_native_research_closing_holdings_not_live_account"
    assert report["benchmarkDefinition"] == "source_native_initial_equal_weight_buy_hold_closing_holdings"
    assert not report["executionAuthorized"]


def test_independent_stage_binds_native_source_metadata_to_the_task_request(bundle, tmp_path):
    import runpy
    from test_native_research_weights import files_for_weights

    manifest = json.loads(bundle.read_bytes())
    files = files_for_weights(manifest['instruments'])
    for name, raw in files.items():
        bundle.with_name(name).write_bytes(raw)
        manifest['files'][name] = hashlib.sha256(raw).hexdigest()
    bundle.write_text(json.dumps(manifest))
    source = json.loads(files['weight-source.json'])
    reference = {field: source[key] for field, key in (
        ('job_id', 'jobId'), ('candidate', 'candidate'), ('currency', 'currency'),
        ('snapshot_at', 'snapshotAt'), ('known_at', 'knownAt'), ('artifacts', 'artifacts'),
        ('source_instruments', 'sourceInstruments'),
    )}
    request = {'kind': 'native_equity_style_risk', 'instruments': manifest['instruments'],
               'end': manifest['end'], 'window': manifest['returnWindow'], 'max_fit_age_days': 7,
               'portfolio_source': reference}
    evaluate = runpy.run_path(str(Path(__file__).resolve().parents[2] / 'scripts/qlib_research/equity.py'))['evaluate_equity_risk']
    cutoff = datetime.fromisoformat(manifest['asOf'])
    report = evaluate(request, bundle, sha(bundle), tmp_path / 'valid', model_known_at=cutoff)
    assert report['portfolioSource']['source']['jobId'] == reference['job_id']
    changed = {**request, 'portfolio_source': {**reference, 'job_id': 'c' * 32}}
    with pytest.raises(ValueError, match='request_portfolio_binding'):
        evaluate(changed, bundle, sha(bundle), tmp_path / 'changed', model_known_at=cutoff)
    with pytest.raises(ValueError, match='unrequested_portfolio_source'):
        evaluate({**request, 'portfolio_source': None}, bundle, sha(bundle), tmp_path / 'missing', model_known_at=cutoff)
    with pytest.raises(ValueError, match='request_scope_binding'):
        evaluate({**request, 'mode': 'recompute'}, bundle, sha(bundle), tmp_path / 'unbound', model_known_at=cutoff)


def test_news_task_with_isolated_news_fixture_runs(bundle, tmp_path):
    """Use received price bytes and explicitly synthetic news only in a temporary directory."""
    from types import SimpleNamespace
    from test_received_news import document, load
    from cortexbacktest.pipelines.news_event_commands import news_equity_command
    doc = document()
    raw = json.dumps(doc).replace('2025-01-05T00:00:00+00:00', '2026-10-20T00:00:00+00:00')
    raw = raw.replace('2025-01-02', '2026-10-19').replace('2025-01-01', '2026-10-01').replace('2025-01-04', '2026-10-23')
    doc = json.loads(raw)
    cutoff = json.loads(bundle.read_bytes())['asOf']
    doc['scope']['as_of'] = cutoff
    doc['events'][0]['evidence']['identity']['asOf'] = cutoff
    stage = tmp_path / 'news-input'
    stage.mkdir()
    events, _ = load(stage, doc)
    shutil.copytree(bundle.parent, stage / 'prices')
    request = dict(kind='native_news_event_study', price_source=dict(source_job_id='isolated-fixture', input_sha256=sha(bundle)),
        news_sha256=sha(stage / 'news.json'), event_ids=events['event_id'].to_list(),
        benchmark_instrument_id=70, event_window=[-2,2], estimation_window=60, estimation_gap=5, model='market')
    request_path = stage / 'request.json'
    request_path.write_text(json.dumps(request))
    run = tmp_path / 'news-run'
    news_equity_command(SimpleNamespace(request=request_path, request_sha256=sha(request_path), output=run))
    report = json.loads((run / 'report.json').read_bytes())
    assert report['status'] == 'available' and len(report['events']) == 1
