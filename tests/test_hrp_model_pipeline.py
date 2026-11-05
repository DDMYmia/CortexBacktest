from datetime import datetime, timedelta, timezone
import numpy as np
import polars as pl
import pytest
from cortexbacktest.adapters.qlib_predictions import _hrp_targets
from cortexbacktest.research.optimization import hierarchical_risk_parity


def data():
    rng = np.random.default_rng(23)
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    values = np.exp(np.cumsum(rng.normal(0, [0.01, 0.02, 0.05], (80, 3)), axis=0)) * 100
    latest = {}
    for n, row in enumerate(values):
        for j, price in enumerate(row):
            latest[str(j + 1), str(n)] = {
                "sessionClose": (start + timedelta(days=n)).isoformat(),
                "instrumentId": str(j + 1),
                "close": float(price),
            }
    dates = [start + timedelta(days=n) for n in range(30, 80)]
    targets = pl.DataFrame(
        {
            "date": dates,
            "INSTRUMENT:1": [0.5] * 50,
            "INSTRUMENT:2": [0.5] * 50,
            "INSTRUMENT:3": [0.0] * 50,
        }
    )
    return latest, targets, values


def test_hrp_bridge_uses_native_allocator_and_selected_assets():
    latest, targets, values = data()
    out = _hrp_targets(targets, latest, 20)
    oracle = hierarchical_risk_parity(
        pl.DataFrame(
            {
                "date": list(range(20)),
                "INSTRUMENT:1": values[11:31, 0] / values[10:30, 0] - 1,
                "INSTRUMENT:2": values[11:31, 1] / values[10:30, 1] - 1,
            }
        )
    )
    assert out["INSTRUMENT:1"][0] == pytest.approx(oracle["INSTRUMENT:1"])
    assert out["INSTRUMENT:3"].sum() == 0
    assert out.select(
        pl.sum_horizontal(pl.exclude("date"))
    ).to_series().to_numpy() == pytest.approx(np.ones(50))
    assert out["INSTRUMENT:1"][0] != 0.5


def test_hrp_future_changes_do_not_change_prior_targets():
    latest, targets, _ = data()
    before = _hrp_targets(targets, latest, 20)
    for row in latest.values():
        if datetime.fromisoformat(row["sessionClose"]) > targets["date"][10]:
            row["close"] *= 3
    assert _hrp_targets(targets, latest, 20).head(11).equals(before.head(11))


def test_hrp_incomplete_history_is_rejected():
    latest, targets, _ = data()
    with pytest.raises(ValueError, match="complete trailing"):
        _hrp_targets(targets, latest, 100)
