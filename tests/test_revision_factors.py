from datetime import datetime, timedelta, timezone

import polars as pl
from polars.testing import assert_frame_equal
import pytest

from cortexbacktest.research import MiningFactorSpec, RevisionPanel, build_revision_factor


def test_invalidation_replaces_old_price_until_a_later_verified_receipt():
    revisions, decisions = history()
    revisions = revisions.with_columns(pl.lit(None, dtype=pl.String).alias("invalid_reason"))
    invalid = revisions.slice(1, 1).with_columns(
        pl.lit(None, dtype=pl.Float64).alias("price"),
        pl.lit("unverified_price_basis").alias("invalid_reason"),
        pl.lit(decisions[2, "date"]).alias("recorded_at"))
    corrected = revisions.slice(1, 1).with_columns(pl.lit(115.).alias("price"),
        pl.lit(decisions[4, "date"]).alias("recorded_at"))
    panel = RevisionPanel(pl.concat([revisions, invalid, corrected]), decisions)
    assert panel.prices(cutoff=decisions[1, "date"])[1, "A"] == 110.
    assert panel.prices(cutoff=decisions[3, "date"])[1, "A"] is None
    assert panel.prices(cutoff=decisions[4, "date"])[1, "A"] == 115.
    assert panel.factor(MiningFactorSpec("momentum", 2))[3, "A"] is None
    assert panel.head(4).fingerprint() != RevisionPanel(revisions, decisions.head(4)).fingerprint()
    assert panel.head(4).fingerprint() == RevisionPanel(pl.concat([revisions, invalid]), decisions.head(4)).fingerprint()


@pytest.mark.parametrize("price,reason", [(None, None), (100., "unverified_price_basis"), (None, "unknown")])
def test_invalidation_requires_an_explicit_supported_reason_and_no_usable_price(price, reason):
    revisions, decisions = history()
    revisions = revisions.with_columns(pl.lit(price, dtype=pl.Float64).alias("price"),
                                       pl.lit(reason, dtype=pl.String).alias("invalid_reason"))
    with pytest.raises(ValueError, match="Invalid revisions"):
        RevisionPanel(revisions, decisions)


def history():
    dates = [datetime(2024, 1, 1, 21, tzinfo=timezone.utc) + timedelta(days=i) for i in range(6)]
    return pl.DataFrame({"date": dates, "symbol": ["A"] * 6, "price": [100., 110., 120., 130., 140., 150.],
                         "recorded_at": dates, "known_at": dates}), pl.DataFrame({"date": dates, "decision_time": dates})


@pytest.mark.parametrize("indicator", ["momentum", "sma_ratio", "rolling_zscore", "rolling_volatility", "risk_adjusted_momentum"])
def test_unrevised_history_matches_native_builder(indicator):
    revisions, decisions = history()
    spec = MiningFactorSpec(indicator, 2)
    expected = spec.build(revisions.select("date", pl.col("price").alias("A")))
    assert_frame_equal(build_revision_factor(spec, revisions, decisions), expected, rel_tol=1e-12, abs_tol=1e-12)


def test_late_revision_changes_future_warmup_but_never_rewrites_past_signals():
    revisions, decisions = history()
    spec = MiningFactorSpec("momentum", 2)
    before = build_revision_factor(spec, revisions, decisions)
    revision = revisions.slice(1, 1).with_columns(pl.lit(220.).alias("price"),
        pl.lit(decisions[3, "date"]).alias("recorded_at"))
    after = build_revision_factor(spec, pl.concat([revisions, revision]), decisions)
    assert after.head(3).equals(before.head(3))
    assert after[3, "A"] == pytest.approx(130 / 220 - 1)
    assert before[3, "A"] == pytest.approx(130 / 110 - 1)


def test_source_availability_and_decision_clock_are_independent_of_receipt():
    revisions, decisions = history()
    late = revisions.with_columns(pl.col("recorded_at") + timedelta(minutes=30))
    spec = MiningFactorSpec("momentum", 2)
    assert build_revision_factor(spec, late, decisions)["A"].null_count() == 6
    after_close = decisions.with_columns(pl.col("decision_time") + timedelta(hours=1))
    assert build_revision_factor(spec, late, after_close)[2, "A"] == pytest.approx(.2)
    unpublished = late.with_columns(pl.col("known_at") + timedelta(days=10))
    assert build_revision_factor(spec, unpublished, after_close)["A"].null_count() == 6


def test_missing_session_is_not_compressed_or_forward_filled():
    revisions, decisions = history()
    incomplete = revisions.filter(pl.col("date") != decisions[1, "date"])
    for indicator in ("sma_ratio", "rolling_volatility"):
        result = build_revision_factor(MiningFactorSpec(indicator, 2), incomplete, decisions)
        assert result[2, "A"] is None
        assert result.height == decisions.height
    # Intermediate missing observations do not invalidate endpoint momentum.
    assert build_revision_factor(MiningFactorSpec("momentum", 2), incomplete, decisions)[2, "A"] == pytest.approx(.2)


@pytest.mark.parametrize("invalid", ["duplicate", "naive", "missing_clock", "price", "early_decision"])
def test_ambiguous_or_unverifiable_history_is_rejected(invalid):
    revisions, decisions = history()
    if invalid == "duplicate":
        revisions = pl.concat([revisions, revisions.head(1)])
    elif invalid == "naive":
        revisions = revisions.with_columns(pl.col("known_at").dt.replace_time_zone(None))
    elif invalid == "missing_clock":
        revisions = revisions.with_columns(pl.lit(None, dtype=pl.Datetime("us", "UTC")).alias("known_at"))
    elif invalid == "price":
        revisions = revisions.with_columns(pl.lit(float("nan")).alias("price"))
    else:
        decisions = decisions.with_columns(pl.col("decision_time") - timedelta(minutes=1))
    with pytest.raises(ValueError):
        build_revision_factor(MiningFactorSpec("momentum", 2), revisions, decisions)


def test_revision_features_feed_existing_research_and_event_execution():
    from cortexbacktest import load_backtest_dataset, ResearchExecutionPipelineConfig

    base, decisions = history()
    revisions = pl.concat([base.with_columns(pl.lit(symbol).alias("symbol"),
        pl.Series("price", [100 * (1 + rate) ** i for i in range(6)]))
        for symbol, rate in zip("ABC", (.01, .02, .03), strict=True)])
    correction = revisions.filter(pl.col("symbol") == "C").slice(1, 1).with_columns(
        (pl.col("price") * 1.1), pl.lit(decisions[3, "date"]).alias("recorded_at"))
    factors = build_revision_factor(MiningFactorSpec("momentum", 2), pl.concat([revisions, correction]), decisions)
    raw = revisions.select("date", "symbol", *(pl.col("price").alias(c) for c in ("open", "high", "low", "close")),
                           pl.lit(1_000_000.).alias("volume"))
    dataset = load_backtest_dataset(bars=raw, factors=factors.unpivot(index="date", variable_name="symbol", value_name="factor"))
    result = dataset.run_research_execution_pipeline(execution_bars=raw,
        config=ResearchExecutionPipelineConfig(horizons=(1,), quantiles=3, top_quantile=1/3, bottom_quantile=None))
    assert result.event.fills
    assert min(fill.timestamp for fill in result.event.fills) == decisions[3, "date"]
    assert factors.height == decisions.height
    assert result.research.target_weights["date"].equals(decisions["date"].slice(2))
