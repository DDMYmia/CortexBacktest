"""Compare bounded fast-path batches with per-decision factors, including revisions."""
from datetime import datetime, timedelta, timezone
import random

import polars as pl
from polars.testing import assert_frame_equal
import pytest

from cortexbacktest.research import MiningFactorSpec, RevisionPanel

START = datetime(2024, 1, 1, 21, tzinfo=timezone.utc)


def history(seed: int, *, late: float, invalid: float, missing: float):
    rng = random.Random(seed)
    dates = [START + timedelta(days=i) for i in range(60)]
    rows = []
    for i, day in enumerate(dates):
        for symbol in ("A", "B", "C", "D"):
            if rng.random() < missing:
                continue
            price = 100 * (1 + rng.uniform(-.03, .03)) ** i
            known = day + timedelta(hours=rng.choice((0, 0, 0, 1)))  # some bars known after the close
            rows.append({"date": day, "symbol": symbol, "price": price, "recorded_at": known, "known_at": known,
                         "invalid_reason": None})
            if rng.random() < late:  # a revision that arrives days later
                when = known + timedelta(days=rng.randint(1, 15), minutes=1)
                bad = rng.random() < invalid
                rows.append({"date": day, "symbol": symbol, "price": None if bad else price * rng.uniform(.9, 1.1),
                             "recorded_at": when, "known_at": known if rng.random() < .5 else when,
                             "invalid_reason": "unverified_price_basis" if bad else None})
    decisions = pl.DataFrame({"date": dates, "decision_time": [d + timedelta(hours=1) for d in dates]})
    return pl.DataFrame(rows, schema={"date": pl.Datetime("us", "UTC"), "symbol": pl.String, "price": pl.Float64,
                                      "recorded_at": pl.Datetime("us", "UTC"), "known_at": pl.Datetime("us", "UTC"),
                                      "invalid_reason": pl.String}), decisions


def per_decision(panel: RevisionPanel, spec: MiningFactorSpec) -> pl.DataFrame:
    return panel.members(pl.concat([panel._factor_at(spec, i) for i in range(panel.decisions.height)]))


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("policy,replay", [("received", False), ("source_publication_replay", False),
                                           ("source_publication_replay", True)])
@pytest.mark.parametrize("indicator", ["momentum", "sma_ratio", "rolling_volatility", "rolling_zscore",
                                       "risk_adjusted_momentum", "rank(momentum)"])
def test_fast_path_equals_the_per_decision_build(seed, policy, replay, indicator):
    revisions, decisions = history(seed, late=.15, invalid=.3, missing=.05)
    replay_as_of = START + timedelta(days=30) if replay else None
    panel = RevisionPanel(revisions, decisions, policy, replay_as_of)
    spec = MiningFactorSpec(indicator, 5)
    expected = (per_decision(panel, spec) if spec.indicator in {"momentum", "sma_ratio", "rolling_volatility",
                "rolling_zscore", "risk_adjusted_momentum"} else None)
    fast = panel.factor(spec)
    if expected is not None:
        assert_frame_equal(fast, expected, check_exact=True)
    assert fast.height == decisions.height and fast["date"].is_sorted()


def test_composites_equal_their_per_decision_leaves():
    revisions, decisions = history(3, late=.2, invalid=.3, missing=.05)
    panel = RevisionPanel(revisions, decisions)
    spec = MiningFactorSpec("add(rank(momentum),rank(sma_ratio))", 5)
    leaves = {name: per_decision(panel, MiningFactorSpec(name, 5)) for name in ("momentum", "sma_ratio")}
    expected = spec.compose(lambda component: leaves[component.indicator])
    assert_frame_equal(panel.factor(spec), expected, check_exact=True)


def test_a_late_revision_forces_the_affected_decisions_back_to_their_own_view():
    revisions, decisions = history(1, late=0., invalid=0., missing=0.)
    late = revisions.filter(pl.col("date") == START + timedelta(days=20), pl.col("symbol") == "A").with_columns(
        (pl.col("price") * 2).alias("price"), (pl.col("recorded_at") + timedelta(days=10)).alias("recorded_at"),
        (pl.col("known_at") + timedelta(days=10)).alias("known_at"))
    panel = RevisionPanel(pl.concat([revisions, late]), decisions)
    settled = panel._settled(6)
    # Decisions 20 to 25 hold day 20 in their window before the revision was visible to them.
    assert not settled[20:26].any() and settled[:20].all() and settled[31:].all()
    assert_frame_equal(panel.factor(MiningFactorSpec("momentum", 5)), per_decision(panel, MiningFactorSpec("momentum", 5)), check_exact=True)


@pytest.mark.parametrize("batch_cells", [1, 200, 1_000_000])
def test_window_batches_equal_the_per_decision_build(monkeypatch, batch_cells):
    from cortexbacktest.research import revisions

    monkeypatch.setattr(revisions, "WINDOW_BATCH_CELLS", batch_cells)
    revisions_frame, decisions = history(5, late=.01, invalid=.3, missing=.05)
    panel = RevisionPanel(revisions_frame, decisions)
    for indicator in ("momentum", "sma_ratio", "rolling_zscore", "rolling_volatility", "risk_adjusted_momentum"):
        for direction in (-1, 1):
            spec = MiningFactorSpec(indicator, 5, direction)
            assert_frame_equal(panel.factor(spec), per_decision(panel, spec), check_exact=True)
