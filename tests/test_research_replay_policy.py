from datetime import datetime, timedelta, timezone
import copy
import json

import pytest

from cortexbacktest.pipelines.mining_job import mining_revision_panels, run_mining_job
from cortexbacktest.pipelines.mining_store import load_mining_result
from cortexbacktest.pipelines.mining_execution import frozen_factor_quantity_changes


def test_replay_preserves_receipts_binds_policy_and_cannot_be_promoted(tmp_path):
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    records = []
    for day in range(120):
        close = start + timedelta(days=day)
        for asset, rate in enumerate((.001, .002, .003, .004)):
            price = 100 * (1 + rate) ** day
            records.append({"instrumentId": str(asset + 1), "sessionClose": close.isoformat(),
                "sourceKnownAt": close.isoformat(), "recordedAt": (close + timedelta(days=365)).isoformat(),
                "availableAt": (close + timedelta(days=365)).isoformat(), "adjustment": "unadjusted",
                "adjClose": price, "close": price, "open": price, "high": price, "low": price, "volume": 1_000_000.})
    original = copy.deepcopy(records)
    spec = {"direction": "momentum", "budget": 2, "seed": 7, "horizon": 1,
            "trainSessions": 30, "testSessions": 30}
    exported = {"settings": {"calendar": "crypto_utc", "inSample": .75, "costBps": 10, "mining": spec},
                "bars": {"sha256": "a" * 64}, "asOf": (close + timedelta(days=400)).isoformat()}
    strict = run_mining_job({"records": records}, exported, tmp_path / "strict")
    assert strict["pit"]["eligibleRecords"] == 0 and not strict["selected"]
    spec["knowledgePolicy"] = "source_publication_replay"
    replay = run_mining_job({"records": records}, exported, tmp_path / "replay")
    assert records == original
    assert replay["pit"]["eligibleRecords"] == 480
    assert replay["pit"]["formalTradingEligible"] is False
    assert replay["selected"] and replay["execution"]["status"] == "evaluated"
    checkpoint = tmp_path / "replay/selection"
    frozen = load_mining_result(checkpoint, manifest_sha256=replay["manifestSha256"])
    assert frozen.settings["knowledge_policy"] == "source_publication_replay"
    _, panel, _, _ = mining_revision_panels({"records": records}, exported)
    assert panel.head(90).knowledge_policy == panel.through(frozen.inner_end).knowledge_policy == panel.knowledge_policy
    strict_panel = type(panel)(panel.revisions, panel.decisions)
    assert strict_panel.fingerprint() != panel.fingerprint()
    with pytest.raises(ValueError, match="historical_replay_not_admitted_to_shadow"):
        frozen_factor_quantity_changes(str(checkpoint), manifest_sha256=replay["manifestSha256"],
            factor=frozen.selected[0].name, revision_panel=panel, as_of=close,
            prices={}, quantities={}, equity=100_000.)
    manifest = json.loads((checkpoint / "manifest.json").read_text())
    assert manifest["settings"]["knowledge_policy"] == "source_publication_replay"
