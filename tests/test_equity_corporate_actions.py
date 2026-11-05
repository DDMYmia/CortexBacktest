"""US-equity execution: raw prices plus the split and dividend ledger, verified like any input."""
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json

import pytest

from cortexbacktest.pipelines.mining_job import load_corporate_actions, run_mining_job

SPLIT_AT = 160  # instrument 4 splits 2-for-1 on this session, inside the simulated (out-of-sample) period


def _study(tmp_path, *, ledger=True, tamper=False):
    days = [d for d in (datetime(2024, 1, 2, 21, tzinfo=timezone.utc) + timedelta(days=i) for i in range(260))
            if d.weekday() < 5][:180]
    records = []
    for index, day in enumerate(days):
        for asset, rate in enumerate((.001, .002, .003, .004)):
            adjusted = 100 * (1 + rate) ** index
            raw = adjusted / 2 if asset == 3 and index >= SPLIT_AT else adjusted
            stamp = day.isoformat()
            records.append({"instrumentId": str(asset + 1), "sessionClose": stamp, "availableAt": stamp,
                            "recordedAt": stamp, "sourceKnownAt": stamp, "adjClose": adjusted, "open": raw,
                            "high": raw, "low": raw, "close": raw, "volume": 1e6, "adjustment": "total_return"})
    settings = {"calendar": "us_equity", "inSample": .75, "costBps": 10, "mining": {
        "direction": "momentum", "budget": 2, "seed": 7, "horizon": 1, "trainSessions": 30, "testSessions": 30,
        "knowledgePolicy": "source_publication_replay", "portfolioMode": "long_only"}}
    exported = {"settings": settings, "bars": {"sha256": "a" * 64}, "asOf": (days[-1] + timedelta(days=1)).isoformat()}
    if ledger:
        document = {"contractVersion": "corporate-actions-v1", "records": [
            {"instrumentId": "4", "actionType": "split", "exDate": days[SPLIT_AT].date().isoformat(), "ratio": 2.0},
            {"instrumentId": "1", "actionType": "cash_dividend", "exDate": days[170].date().isoformat(),
             "amount": 0.25, "payDate": None}]}
        content = json.dumps(document).encode()
        (tmp_path / "corporate-actions.json").write_bytes(content)
        exported["corporateActions"] = {"path": str(tmp_path / "corporate-actions.json"),
                                        "sha256": sha256(b"x" if tamper else content).hexdigest()}
    return run_mining_job({"records": records}, exported, tmp_path / "study")


def test_an_equity_study_executes_on_raw_prices_with_its_ledger(tmp_path):
    result = _study(tmp_path)
    assert result["execution"]["status"] == "evaluated" and result["selected"]
    assert result["inputs"]["corporateActions"] == sha256((tmp_path / "corporate-actions.json").read_bytes()).hexdigest()
    # Prices only rise: a long book never loses half its value at the split once the split is applied.
    for item in result["execution"]["results"]:
        assert item["report"]["summary"]["event_max_drawdown"] > -0.2
        # Fills and the split together account for final equity.
        assert item["report"]["event"]["pnl_reconciliation"]["reconciliation_error"] == pytest.approx(0.0, abs=1e-6)


def test_without_a_ledger_an_equity_study_refuses_to_simulate(tmp_path):
    assert _study(tmp_path, ledger=False)["execution"]["status"] == "requires_corporate_actions"


def test_a_ledger_must_match_its_digest(tmp_path):
    with pytest.raises(ValueError, match="digest"):
        _study(tmp_path, tamper=True)


def test_ledger_records_become_engine_actions(tmp_path):
    content = json.dumps({"contractVersion": "corporate-actions-v1", "records": [
        {"instrumentId": "7", "actionType": "cash_dividend", "exDate": "2024-03-01", "amount": 0.5, "payDate": "2024-03-15"}]}).encode()
    (tmp_path / "ledger.json").write_bytes(content)
    [action], _ = load_corporate_actions({"corporateActions": {"path": str(tmp_path / "ledger.json"),
                                                               "sha256": sha256(content).hexdigest()}})
    assert (action.symbol, str(action.ex_date), action.amount, str(action.pay_date)) == ("INSTRUMENT:7", "2024-03-01", 0.5, "2024-03-15")
