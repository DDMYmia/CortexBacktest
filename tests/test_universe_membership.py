"""Point-in-time universe: factor values exist only while a symbol is an index member."""
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json

import polars as pl
import pytest

from cortexbacktest.pipelines.mining_job import run_mining_job
from cortexbacktest.research import MiningFactorSpec, RevisionPanel

DAYS = [datetime(2024, 1, 1, 21, tzinfo=timezone.utc) + timedelta(days=i) for i in range(8)]


def panel(membership=None):
    rows = [{"date": d, "symbol": s, "price": 100. * (1 + r) ** i, "recorded_at": d, "known_at": d}
            for i, d in enumerate(DAYS) for s, r in (("A", .01), ("B", .02), ("C", .03))]
    return RevisionPanel(pl.DataFrame(rows), pl.DataFrame({"date": DAYS, "decision_time": DAYS}),
                         membership=membership)


def stretches(*rows):
    return pl.DataFrame([{"symbol": s, "start": DAYS[a], "end": DAYS[b]} for s, a, b in rows])


def test_a_factor_is_null_outside_membership_but_its_history_still_warms_up():
    spec = MiningFactorSpec("momentum", 2)
    full = panel().factor(spec)
    # C joins on day 5; B leaves after day 3 and returns on day 6; A is never a member.
    masked = panel(stretches(("B", 0, 3), ("B", 6, 7), ("C", 5, 7))).factor(spec)
    assert masked["A"].null_count() == masked.height
    assert masked["B"].to_list()[4:6] == [None, None]
    assert masked["B"][6] == full["B"][6]
    # C's first member day has a value: the lookback used prices from before it joined.
    assert masked["C"][5] == full["C"][5] and masked["C"][4] is None


def test_a_cross_sectional_rank_compares_members_only():
    spec = MiningFactorSpec("rank(momentum)", 2)
    two = panel(stretches(("A", 0, 7), ("B", 0, 7))).factor(spec).row(5, named=True)
    three = panel(stretches(("A", 0, 7), ("B", 0, 7), ("C", 0, 7))).factor(spec).row(5, named=True)
    # C, the strongest, is not a member: B tops a two-name ranking but sits in the middle of three.
    assert two["C"] is None and two["B"] > two["A"]
    assert two["B"] != three["B"]


def test_membership_is_part_of_the_fingerprint_and_survives_head():
    members = stretches(("A", 0, 7), ("B", 0, 3))
    assert panel(members).head(5).membership is not None
    assert panel(members).head(5).fingerprint() != panel().head(5).fingerprint()
    assert panel(members).head(5).fingerprint() != panel(stretches(("A", 0, 7))).head(5).fingerprint()


def test_reversed_or_malformed_membership_is_refused():
    with pytest.raises(ValueError, match="reversed"):
        panel(stretches(("A", 5, 2)))
    with pytest.raises(ValueError, match="symbol, start and end"):
        panel(pl.DataFrame({"symbol": ["A"], "start": [DAYS[0]]}))


def _study(tmp_path, *, universe="sp500", tamper=False, membership_rows=None):
    dates = [datetime(2024, 1, 1, 21, tzinfo=timezone.utc) + timedelta(days=i) for i in range(160)]
    records = []
    for index, day in enumerate(dates):
        for asset, rate in enumerate((.001, .002, .003, .004, .005)):
            price = 100 * (1 + rate) ** index
            records.append({"instrumentId": str(asset + 1), "sessionClose": day.isoformat(),
                            "availableAt": day.isoformat(), "recordedAt": day.isoformat(),
                            "sourceKnownAt": day.isoformat(), "adjClose": price, "open": price, "high": price,
                            "low": price, "close": price, "volume": 1e6, "adjustment": "unadjusted"})
    # Instrument 5, the strongest, joins only for the last 20 sessions.
    rows = membership_rows or [{"instrumentId": str(i), "start": dates[0].isoformat(), "end": dates[-1].isoformat()}
                               for i in range(1, 5)] + [
        {"instrumentId": "5", "start": dates[140].isoformat(), "end": dates[-1].isoformat()}]
    content = json.dumps({"contractVersion": "index-membership-v1", "universe": "sp500", "index": "SPY",
                          "records": rows}).encode()
    path = tmp_path / "membership.json"
    path.write_bytes(content)
    settings = {"calendar": "crypto_utc", "inSample": .75, "costBps": 10, "mining": {
        "direction": "momentum", "budget": 2, "seed": 7, "horizon": 1, "trainSessions": 30, "testSessions": 30,
        **({"universe": universe} if universe else {})}}
    exported = {"settings": settings, "bars": {"sha256": "a" * 64}, "asOf": dates[-1].isoformat(),
                "membership": {"path": str(path), "sha256": sha256(b"x" if tamper else content).hexdigest()}}
    return run_mining_job({"records": records}, exported, tmp_path / "study")


def test_a_universe_study_binds_its_membership_and_says_so(tmp_path):
    result = _study(tmp_path)
    membership = json.loads((tmp_path / "membership.json").read_text())
    assert result["inputs"]["membership"] == sha256((tmp_path / "membership.json").read_bytes()).hexdigest()
    assert result["pit"]["universePolicy"] == "point_in_time_index_membership"
    assert result["pit"]["membership"]["instruments"] == 5 and result["pit"]["membership"]["index"] == "SPY"
    assert membership["universe"] == "sp500"


def test_a_membership_file_must_match_its_digest_and_its_study(tmp_path):
    with pytest.raises(ValueError, match="digest"):
        _study(tmp_path, tamper=True)
    with pytest.raises(ValueError, match="only a universe study"):
        (tmp_path / "fixed").mkdir()
        _study(tmp_path / "fixed", universe=None)
