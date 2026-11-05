from datetime import UTC, datetime

import pandas as pd
import pytest

from cortexbacktest import (
    DatasetProvenance,
    load_backtest_dataset,
    load_backtest_dataset_dir,
    save_backtest_dataset,
)


def _bars():
    return pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-03"],
            "symbol": ["A", "A"],
            "open": [10.0, 11.0],
            "high": [10.0, 11.0],
            "low": [10.0, 11.0],
            "close": [10.0, 11.0],
            "adj_close": [10.0, 11.0],
            "volume": [100, 100],
        }
    )


def _factors():
    return pd.DataFrame({"date": ["2024-01-02"], "symbol": ["A"], "factor": [1.0]})


def test_dataset_provenance_and_component_hashes_round_trip(tmp_path):
    provenance = DatasetProvenance(
        "vendor", "us_equities", datetime(2024, 1, 4, tzinfo=UTC), "2024.01", "r2"
    )
    dataset = load_backtest_dataset(bars=_bars(), factors=_factors(), provenance=provenance)
    output = save_backtest_dataset(dataset, tmp_path / "dataset")

    loaded = load_backtest_dataset_dir(output)

    assert loaded.provenance == provenance


def test_dataset_hash_verification_rejects_tampered_component(tmp_path):
    dataset = load_backtest_dataset(bars=_bars(), factors=_factors())
    output = save_backtest_dataset(dataset, tmp_path / "dataset")
    (output / "bars.parquet").write_text("tampered", encoding="utf-8")

    with pytest.raises(ValueError, match="hash mismatch"):
        load_backtest_dataset_dir(output)
