"""US-equity decisions after the session's bars are known, filled at the next open."""
from datetime import datetime, timedelta, timezone

import pytest

from cortexbacktest.pipelines.mining_job import mining_revision_panels, run_mining_job


def _study(clock=None, *, calendar="us_equity", delay=0):
    # Synthetic weekdays start in June with fixed 20:00/21:00 UTC stamps.
    # Later rows cross DST; this fixture does not validate real session hours.
    days = [datetime(2024, 6, 3, tzinfo=timezone.utc) + timedelta(days=i) for i in range(220)]
    days = [d for d in days if d.weekday() < 5][:150]
    records = []
    for index, day in enumerate(days):
        close_at, known_at = day.replace(hour=20), day.replace(hour=21)
        for asset, rate in enumerate((.001, .002, .003, .004)):
            close = 100 * (1 + rate) ** index
            records.append({"instrumentId": str(asset + 1), "sessionClose": close_at.isoformat(),
                            "availableAt": known_at.isoformat(), "recordedAt": known_at.isoformat(),
                            "sourceKnownAt": known_at.isoformat(), "adjClose": close / 2, "open": close * .99,
                            "high": close, "low": close * .98, "close": close, "volume": 1e6,
                            "adjustment": "total_return"})
    mining = {"direction": "momentum", "budget": 2, "seed": 7, "horizon": 1, "trainSessions": 30, "testSessions": 30,
              "knowledgePolicy": "source_publication_replay", **({"decisionClock": clock} if clock else {}),
              **({"decisionDelayMinutes": delay} if delay else {})}
    exported = {"settings": {"calendar": calendar, "inSample": .75, "costBps": 10, "mining": mining},
                "bars": {"sha256": "a" * 64}, "asOf": (days[-1] + timedelta(days=1)).isoformat()}
    return {"records": records}, exported


def test_decisions_wait_for_the_bars_and_trade_at_the_next_new_york_open():
    bars, exported = _study("after_bar_known")
    _, panel, execution, _ = mining_revision_panels(bars, exported)
    first = panel.decisions.row(0, named=True)
    assert first["date"].hour == 20 and first["decision_time"].hour == 21
    # 09:30 New York in June is 13:30 UTC, on the same session date.
    assert execution.decisions[0, "date"] == first["date"].replace(hour=13, minute=30)
    assert execution.decisions["decision_time"].equals(panel.decisions["decision_time"])
    # Opens are put on the research close's basis: here adjClose is half the close.
    assert execution.revisions.filter(execution.revisions["symbol"] == "INSTRUMENT:1")[0, "price"] == pytest.approx(49.5)


def test_every_summer_bar_is_eligible_after_it_is_known_and_none_at_the_close(tmp_path):
    bars, exported = _study("after_bar_known")
    after = run_mining_job(bars, exported, tmp_path / "after")
    assert after["pit"]["decisionPolicy"] == "after_bar_known_next_open"
    assert after["pit"]["eligibleRecords"] == after["pit"]["records"]
    bars, exported = _study()
    close = run_mining_job(bars, exported, tmp_path / "close")
    assert close["pit"]["decisionPolicy"] == "session_close" and close["pit"]["eligibleRecords"] == 0


@pytest.mark.parametrize("calendar,delay", [("crypto_utc", 0), ("us_equity", 60)])
def test_the_clock_is_only_for_undelayed_us_equity_sessions(calendar, delay):
    bars, exported = _study("after_bar_known", calendar=calendar, delay=delay)
    with pytest.raises(ValueError):
        mining_revision_panels(bars, exported)
