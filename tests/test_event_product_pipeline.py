from datetime import datetime, timedelta, timezone
import json

import numpy as np
import pytest

from cortexbacktest.adapters.qlib_predictions import _event_study_report


def panel():
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    market = np.sin(np.arange(129) * 0.7) / 100
    asset = 0.001 + 2 * market
    asset[68:71] += 0.03
    values = 100 * np.vstack([np.ones(2), np.cumprod(1 + np.column_stack([asset, market]), axis=0)])
    latest = {
        (str(j + 1), str(i)): {
            "instrumentId": str(j + 1),
            "sessionClose": (start + timedelta(days=i)).isoformat(),
            "close": float(v),
        }
        for i, row in enumerate(values)
        for j, v in enumerate(row)
    }
    event = {
        "event_id": "e1",
        "instrument_id": 1,
        "event_at": (start + timedelta(days=70)).isoformat(),
        "available_at": (start + timedelta(days=70)).isoformat(),
    }
    config = {
        "benchmark_instrument_id": 2,
        "events": [event],
        "event_window": [-1, 1],
        "estimation_window": 40,
        "estimation_gap": 5,
        "model": "market",
    }
    return latest, config, start + timedelta(days=129)


def test_event_bridge_matches_independent_analytic_shock_and_binds_input():
    result = _event_study_report(*panel())
    assert result["status"] == "available" and result["events"][0]["car"] == pytest.approx(0.09)
    assert result["events"][0]["alpha"] == pytest.approx(0.001)
    assert result["events"][0]["beta"] == pytest.approx(2.0)
    assert result["averageReturns"][-1]["caar"] == pytest.approx(0.09)
    assert result["eventEvidence"] == "request_supplied_clocks_not_received_news_receipts"
    assert len(result["inputEventsSha256"]) == 64 and not result["qualificationIssued"]
    json.dumps(result, allow_nan=False)


def test_late_availability_anchors_after_receipt_and_future_prices_cannot_change_fit():
    latest, config, cutoff = panel()
    before = _event_study_report(latest, config, cutoff)
    for (_, i), row in latest.items():
        if int(i) > 80:
            row["close"] *= 3
    assert _event_study_report(latest, config, cutoff) == before
    config["events"][0]["available_at"] = "2025-03-14T06:00:00Z"
    shifted = _event_study_report(latest, config, cutoff)
    assert shifted["events"][0]["anchor"] == "2025-03-15T00:00:00+00:00"


def test_overlap_suppresses_iid_inference_and_empty_events_are_unavailable():
    latest, config, cutoff = panel()
    config["events"].append({**config["events"][0], "event_id": "e2"})
    result = _event_study_report(latest, config, cutoff)
    assert result["clusters"][0]["eventIds"] == ["e1", "e2"]
    assert result["inference"]["pValue"] is None and "overlapping" in result["inference"]["reason"]
    config["events"] = []
    empty = _event_study_report(latest, config, cutoff)
    assert empty["status"] == "unavailable" and empty["reason"] == "no_accepted_events"


def test_future_and_incomplete_event_windows_keep_drop_reasons():
    latest, config, cutoff = panel()
    result = _event_study_report(latest, config, datetime(2025, 3, 12, tzinfo=timezone.utc))
    assert result["dropped"][0]["reason"] == "event_window_incomplete_at_cutoff"
    result = _event_study_report(latest, config, datetime(2025, 3, 10, tzinfo=timezone.utc))
    assert result["dropped"][0]["reason"] == "not_available_at_cutoff"


@pytest.mark.parametrize(
    "patch",
    [
        {"benchmark_instrument_id": True},
        {"benchmark_instrument_id": 1},
        {"event_window": [True, 1]},
        {"event_window": [2, -2]},
        {"estimation_window": 29},
        {"extra": 1},
    ],
)
def test_invalid_event_settings_reject(patch):
    latest, config, cutoff = panel()
    with pytest.raises(ValueError):
        _event_study_report(latest, {**config, **patch}, cutoff)


@pytest.mark.parametrize(
    "patch",
    [
        {"available_at": "2025-03-12T00:00:00"},
        {"available_at": True},
        {"instrument_id": 3},
        {"available_at": "2025-03-01T00:00:00Z"},
    ],
)
def test_invalid_event_identity_or_clocks_reject(patch):
    latest, config, cutoff = panel()
    config["events"][0].update(patch)
    with pytest.raises(ValueError):
        _event_study_report(latest, config, cutoff)
