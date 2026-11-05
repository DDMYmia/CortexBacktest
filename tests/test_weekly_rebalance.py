"""Weekly rebalance: values held from each week's first decision, in research and in a live decision."""
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import polars as pl
from polars.testing import assert_frame_equal

from cortexbacktest.research import MiningFactorSpec, RevisionPanel

# Thursday 2024-01-04 onward, daily closes at 00:00 UTC (a crypto day).
DAYS = [datetime(2024, 1, 4, tzinfo=timezone.utc) + timedelta(days=i) for i in range(40)]


def panel(rebalance="weekly"):
    rows = [{"date": d, "symbol": s, "price": 100. * (1 + r) ** i + (i % 3), "recorded_at": d, "known_at": d}
            for i, d in enumerate(DAYS) for s, r in (("A", .01), ("B", -.01), ("C", .02))]
    return RevisionPanel(pl.DataFrame(rows), pl.DataFrame({"date": DAYS, "decision_time": DAYS}), rebalance=rebalance)


def test_each_value_is_its_weeks_first_decision():
    spec = MiningFactorSpec("rank(momentum)", 3)
    daily, weekly = panel("daily").factor(spec), panel().factor(spec)
    weeks = weekly["date"].dt.truncate("1w")
    for week in weeks.unique():
        rows = weekly.filter(weeks == week)
        first = daily.filter(pl.col("date") == rows["date"].min())
        assert all(rows.drop("date").row(i) == first.drop("date").row(0) for i in range(rows.height))
    # Raw momentum changes every day; held, it changes at most once a week.
    raw_daily = panel("daily").factor(MiningFactorSpec("momentum", 3)).drop_nulls()
    raw_weekly = panel().factor(MiningFactorSpec("momentum", 3)).drop_nulls()
    assert raw_weekly.drop("date").n_unique() <= raw_weekly["date"].dt.truncate("1w").n_unique()
    assert raw_daily.drop("date").n_unique() > 4 * raw_weekly.drop("date").n_unique()


def test_a_live_decision_holds_what_research_held_on_that_day():
    spec = MiningFactorSpec("momentum", 3)
    research = panel().factor(spec)
    for size in (12, 15, 20, 33):
        live = panel().head(size).factor(spec, last_only=True)
        assert_frame_equal(live, research.slice(size - 1, 1), check_exact=True)


def test_rebalance_is_part_of_the_fingerprint_and_survives_head():
    assert panel().head(10).rebalance == "weekly"
    assert panel().fingerprint() != panel("daily").fingerprint()
    assert replace(panel(), rebalance="daily").fingerprint() == panel("daily").fingerprint()
