from datetime import datetime, timedelta, timezone
import hashlib
import json

import numpy as np
import polars as pl
import pytest

from cortexbacktest.research import event_study
from cortexbacktest.adapters.news_events import load_news_replay_events


def fixture():
    dates = [datetime(2025, 1, 1, tzinfo=timezone.utc) + timedelta(days=n) for n in range(130)]
    m = np.sin(np.arange(130) * .7) / 100
    asset = .001 + 2 * m
    asset[69:72] += .03
    returns = pl.DataFrame({"date": dates, "A": asset, "B": asset + .001})
    market = pl.DataFrame({"date": dates, "market": m})
    events = pl.DataFrame({"event_id": ["e1"], "symbol": ["A"],
                           "event_at": [dates[70]], "available_at": [dates[70]]})
    return dates, returns, market, events


def study(returns, market, events, cutoff=None, **kwargs):
    return event_study(returns, market, events,
        as_of=cutoff or returns["date"][-1], estimation_window=40, estimation_gap=5,
        event_window=kwargs.pop("event_window", (-1, 1)), **kwargs)


def test_analytic_ar_car_caar_and_isolated_fit():
    dates, returns, market, events = fixture()
    result = study(returns, market, events)
    row = result.events[0]
    assert row["alpha"] == pytest.approx(.001)
    assert row["beta"] == pytest.approx(2.)
    assert row["car"] == pytest.approx(.09)
    assert result.abnormal_returns["ar"].to_list() == pytest.approx([.03] * 3)
    assert result.average_returns["caar"][-1] == pytest.approx(.09)
    assert datetime.fromisoformat(row["estimationEnd"]) < dates[69] - timedelta(days=5)
    assert row["standardizedCar"] is None
    json.dumps(result.to_dict(), allow_nan=False)
    changed = returns.with_columns(pl.when(pl.col("date") >= dates[69])
        .then(pl.col("A") + 100).otherwise(pl.col("A")).alias("A"))
    later = study(changed, market, events)
    assert later.events[0]["alpha"] == row["alpha"]
    assert later.events[0]["beta"] == row["beta"]


def test_late_receipt_moves_anchor_forward_not_to_publication():
    dates, returns, market, events = fixture()
    events = events.with_columns(pl.lit(dates[71] + timedelta(hours=6)).alias("available_at"))
    result = study(returns, market, events)
    assert result.events[0]["anchor"] == dates[72].isoformat()
    assert result.events[0]["eventAt"] == dates[70].isoformat()


def test_future_events_and_incomplete_windows_are_reported():
    dates, returns, market, events = fixture()
    assert study(returns, market, events, dates[69]).dropped[0]["reason"] == "not_available_at_cutoff"
    result = study(returns, market, events, dates[70])
    assert result.dropped[0]["reason"] == "event_window_incomplete_at_cutoff"
    assert result.average_returns.is_empty()
    json.dumps(result.to_dict(), allow_nan=False)


def test_positive_window_never_moves_estimation_past_anchor():
    dates, returns, market, events = fixture()
    result = study(returns, market, events, event_window=(2, 4))
    assert datetime.fromisoformat(result.events[0]["estimationEnd"]) < dates[70]


def test_overlapping_windows_form_clusters_and_disable_iid_test():
    dates, returns, market, events = fixture()
    second = events.with_columns(pl.lit("e2").alias("event_id"), pl.lit("B").alias("symbol"),
                                  pl.lit(dates[72]).alias("available_at"))
    result = study(returns, market, pl.concat([events, second]))
    assert result.clusters[0]["eventIds"] == ["e1", "e2"]
    assert result.inference["pValue"] is None
    assert "overlapping" in result.inference["reason"]


@pytest.mark.parametrize("kind", ["naive", "duplicate", "availability", "benchmark"])
def test_invalid_clock_identity_and_benchmark_rejected(kind):
    dates, returns, market, events = fixture()
    if kind == "naive":
        returns = returns.with_columns(pl.col("date").dt.replace_time_zone(None))
    elif kind == "duplicate":
        events = pl.concat([events, events])
    elif kind == "availability":
        events = events.with_columns(pl.lit(dates[68]).alias("available_at"))
    else:
        market = market.slice(1)
    with pytest.raises(ValueError):
        study(returns, market, events)


def test_nonfinite_windows_drop_instead_of_imputation():
    dates, returns, market, events = fixture()
    returns = returns.with_columns(pl.when(pl.col("date") == dates[50])
        .then(float("nan")).otherwise(pl.col("A")).alias("A"))
    assert study(returns, market, events).dropped[0]["reason"] == "nonfinite_estimation_or_event_returns"


def test_nonoverlapping_event_statistics_are_finite():
    dates, returns, market, events = fixture()
    second = events.with_columns(pl.lit("e2").alias("event_id"), pl.lit("B").alias("symbol"),
        pl.lit(dates[100]).alias("event_at"), pl.lit(dates[100]).alias("available_at"))
    result = study(returns, market, pl.concat([events, second]))
    assert result.inference["reason"] is None
    assert 0 <= result.inference["pValue"] <= 1


def test_news_adapter_checks_rehashed_receipt_clocks_and_scope(tmp_path):
    clock = "2025-03-12T00:00:00+00:00"
    doc = {"scope": {"instrument_id": 42, "as_of": "2025-06-01T00:00:00+00:00"},
        "gaps": [], "executionAuthorized": False, "events": [{"kind": "news", "articleId": 8,
        "publishedAt": clock, "receivedAt": clock, "entityReceivedAt": clock,
        "identityReceivedAt": clock, "contentReceivedAt": clock, "availableAt": clock}]}
    def load():
        raw = json.dumps(doc).encode()
        path = tmp_path / "replay.json"
        path.write_bytes(raw)
        return load_news_replay_events(path, sha256=hashlib.sha256(raw).hexdigest())
    events, meta = load()
    assert events["symbol"].to_list() == ["INSTRUMENT:42"]
    assert meta["identityPolicy"] == "upstream_explicit_scope_not_ticker_guessing"
    doc["events"][0]["receivedAt"] = "2025-03-13T00:00:00+00:00"
    with pytest.raises(ValueError, match="clock"):
        load()


def test_unknown_publication_is_preserved_without_inventing_a_clock():
    _, returns, market, events = fixture()
    events = events.with_columns(pl.lit(None, dtype=pl.Datetime("us", "UTC")).alias("event_at"))
    result = study(returns, market, events)
    assert result.events[0]["eventAt"] is None
    assert result.events[0]["availableAt"] is not None
