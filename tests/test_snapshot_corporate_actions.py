from datetime import date, datetime, timedelta
import json

import polars as pl
import pytest

from cortexbacktest import CorporateAction, CorporateActionType, load_backtest_dataset, load_backtest_dataset_dir
from cortexbacktest.research.frames import forward_returns_matrix


@pytest.mark.parametrize("format", ["csv", "parquet"])
@pytest.mark.parametrize("amount", [0.0, 30.0])
def test_snapshot_preserves_delisting_economics(tmp_path, format, amount):
    days = [datetime(2024, 1, 1) + timedelta(days=n) for n in range(5)]
    bars = pl.DataFrame([
        {"date": day, "symbol": symbol, "adj_close": 100.0}
        for n, day in enumerate(days) for symbol in ["GOOD", "DEAD"]
        if symbol == "GOOD" or n < 3
    ])
    factors = pl.DataFrame({"date": [days[0]], "symbol": ["DEAD"], "factor": [-1.0]})
    actions = (a for a in [CorporateAction("DEAD", date(2024, 1, 4), CorporateActionType.DELISTING, amount=amount)])
    original = load_backtest_dataset(bars=bars, factors=factors, corporate_actions=actions)
    path = original.save(tmp_path / "snapshot", format=format)
    loaded = load_backtest_dataset_dir(path)
    assert loaded.corporate_actions == original.corporate_actions
    assert loaded.prices.equals(original.prices)
    assert loaded.returns.equals(original.returns)
    terminal = forward_returns_matrix(loaded.prices, 1).filter(pl.col("date") == days[2])["DEAD"].item()
    assert terminal == pytest.approx(amount / 100 - 1)
    (path / "corporate_actions.json").write_text("[]")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_backtest_dataset_dir(path)
    manifest = json.loads((path / "manifest.json").read_text())
    manifest["sha256"].pop("corporate_actions.json")
    (path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="missing component hashes"):
        load_backtest_dataset_dir(path)
