from datetime import datetime

import polars as pl
import pytest

from cortexbacktest import (
    Bar,
    Order,
    Side,
    Strategy,
    load_backtest_dataset,
    load_backtest_dataset_dir,
    save_backtest_dataset,
)
from cortexbacktest.adapters import build_export_dataset
from cortexbacktest.datasets import bars_to_events, normalize_bars, validate_bars


def _bars() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "date": [
                "2024-01-01",
                "2024-01-01",
                "2024-01-02",
                "2024-01-02",
                "2024-01-03",
                "2024-01-03",
            ],
            "symbol": ["A", "B", "A", "B", "A", "B"],
            "adj_close": [10.0, 20.0, 11.0, 19.0, 12.0, 18.0],
        }
    )


def _factors() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"],
            "symbol": ["A", "B", "A", "B"],
            "factor": [1.0, -1.0, 1.0, -1.0],
        }
    )


def _cell(frame: pl.DataFrame, day: int, symbol: str):
    return frame.filter(pl.col("date") == datetime(2024, 1, day)).row(0, named=True)[symbol]


def test_load_backtest_dataset_from_dataframes():
    dataset = load_backtest_dataset(bars=_bars(), factors=_factors())

    # Wide Polars matrices carry a date column plus one column per symbol.
    assert dataset.prices.shape == (3, 3)
    assert dataset.factor_values.shape == (2, 3)
    assert dataset.returns.shape == (3, 3)
    assert dataset.calendar.contains("2024-01-02")
    report = dataset.run_factor_research(horizons=(1,), quantiles=2)
    assert report.summary().filter(pl.col("horizon") == 1)["mean_rank_ic"].item() > 0


def test_load_backtest_dataset_from_csv_with_membership(tmp_path):
    bars_path = tmp_path / "bars.csv"
    factors_path = tmp_path / "factors.csv"
    membership_path = tmp_path / "membership.csv"
    _bars().write_csv(bars_path)
    _factors().write_csv(factors_path)
    pl.DataFrame(
        {"symbol": ["A", "B"], "start": ["2024-01-01", "2024-01-02"], "end": [None, None]}
    ).write_csv(membership_path)

    dataset = load_backtest_dataset(
        bars=bars_path, factors=factors_path, membership=membership_path
    )

    assert dataset.universe is not None
    assert _cell(dataset.universe.mask, 1, "A") is True
    assert _cell(dataset.universe.mask, 1, "B") is False


def test_save_and_load_backtest_dataset_directory_round_trips_csv(tmp_path):
    membership = pl.DataFrame(
        {"symbol": ["A", "B"], "start": ["2024-01-01", "2024-01-02"], "end": [None, None]}
    )
    dataset = load_backtest_dataset(bars=_bars(), factors=_factors(), membership=membership)
    output_dir = tmp_path / "dataset"

    saved = save_backtest_dataset(dataset, output_dir, format="csv")
    loaded = load_backtest_dataset_dir(saved)

    assert (saved / "manifest.json").exists()
    assert loaded.bars.shape == dataset.bars.shape
    assert loaded.factors.shape == dataset.factors.shape
    assert loaded.universe is not None
    assert loaded.universe.mask.equals(dataset.universe.mask)


def test_save_and_load_round_trips_source_column_names(tmp_path):
    # A dataset built from source columns that are not the defaults still has
    # to reload from its directory: the manifest carries the column names.
    bars = _bars().rename({"date": "day", "symbol": "ticker", "adj_close": "close"})
    factors = _factors().rename({"date": "day", "symbol": "ticker", "factor": "smart_money_score"})
    dataset = load_backtest_dataset(
        bars=bars,
        factors=factors,
        date_col="day",
        symbol_col="ticker",
        price_col="close",
        factor_col="smart_money_score",
    )

    loaded = load_backtest_dataset_dir(save_backtest_dataset(dataset, tmp_path / "dataset"))

    assert loaded.prices.equals(dataset.prices)
    assert loaded.factor_values.equals(dataset.factor_values)


def test_dataset_save_method_round_trips(tmp_path):
    dataset = load_backtest_dataset(bars=_bars(), factors=_factors())

    loaded = load_backtest_dataset_dir(dataset.save(tmp_path / "dataset"))

    assert loaded.prices.equals(dataset.prices)
    assert loaded.factor_values.equals(dataset.factor_values)


def test_build_export_dataset_normalizes_export_columns():
    bars = _bars().rename({"date": "timestamp", "symbol": "ticker", "adj_close": "adjClose"})
    factors = _factors().rename(
        {"date": "analysis_timestamp", "symbol": "ticker", "factor": "smart_money_score"}
    )

    dataset = build_export_dataset(
        bars, factors, factor_name="smart_money_score", research_mode="exploration"
    )

    assert dataset.prices.columns == ["date", "A", "B"]
    assert _cell(dataset.factor_values, 1, "A") == 1.0


def test_normalize_bars_fills_ohlc_from_adjusted_close():
    bars = normalize_bars(_bars())

    assert bars.columns == ["date", "symbol", "open", "high", "low", "close", "volume", "adj_close"]
    first = bars.row(0, named=True)
    assert first["open"] == first["high"] == first["low"] == first["close"] == 10
    assert validate_bars(bars).ok


def test_validate_bars_rejects_duplicates_and_non_positive_prices():
    bars = normalize_bars(_bars())
    bad = pl.concat([bars, bars[0:1]]).with_columns(
        pl.when(pl.int_range(pl.len()) == 0).then(0.0).otherwise(pl.col("close")).alias("close")
    )

    report = validate_bars(bad)

    assert not report.ok
    assert any("duplicate date/symbol" in error for error in report.errors)
    assert any("close contains" in error for error in report.errors)


def test_bars_to_events_returns_sorted_bar_contracts():
    bars = bars_to_events(_bars().reverse())

    assert [bar.symbol for bar in bars[:2]] == ["A", "B"]
    assert isinstance(bars[0], Bar)
    assert bars[0].close == 10
    assert bars[0].return_close == 10


def test_bars_to_events_preserves_optional_vwap():
    source = _bars().with_columns(pl.Series("vwap", [10.2, 20.2, 11.2, 19.2, 12.2, 18.2]))

    bars = bars_to_events(source)

    assert bars[0].vwap == 10.2


class DatasetBuyOnceStrategy(Strategy):
    def __init__(self) -> None:
        self.bought = False

    def on_bar(self, bar: Bar) -> list[Order]:
        if self.bought or bar.symbol != "A":
            return []
        self.bought = True
        return [Order(symbol=bar.symbol, timestamp=bar.timestamp, side=Side.BUY, quantity=10)]


def test_dataset_can_run_event_backtest():
    dataset = load_backtest_dataset(bars=_bars(), factors=_factors())

    result = dataset.run_event_backtest(DatasetBuyOnceStrategy())

    assert len(result.fills) == 1
    assert result.fills[0].symbol == "A"
    assert result.summary()["final_equity"] > 100_000


def test_load_backtest_dataset_rejects_bad_bar_data():
    bars = _bars().with_columns(
        pl.when(pl.int_range(pl.len()) == 0)
        .then(-1.0)
        .otherwise(pl.col("adj_close"))
        .alias("adj_close")
    )

    with pytest.raises(ValueError, match="adj_close contains"):
        load_backtest_dataset(bars=bars, factors=_factors())


@pytest.mark.parametrize("format", ["csv", "parquet"])
def test_legacy_backend_uses_polars_for_snapshot_io(tmp_path, monkeypatch, format):
    import pandas as pd
    from cortexbacktest.data import backend
    from cortexbacktest.datasets import load_target_weights

    def reject_pandas(*args, **kwargs):
        raise AssertionError("Dataset I/O must not enter pandas")

    monkeypatch.setattr(pd, "read_csv", reject_pandas)
    monkeypatch.setattr(pd, "read_parquet", reject_pandas)
    monkeypatch.setattr(backend, "to_pandas", reject_pandas)
    monkeypatch.setattr(pl.DataFrame, "to_pandas", reject_pandas)
    bars, factors = tmp_path / "input-bars.csv", tmp_path / "input-factors.csv"
    _bars().write_csv(bars)
    _factors().write_csv(factors)
    dataset = load_backtest_dataset(bars=bars, factors=factors)
    path = save_backtest_dataset(dataset, tmp_path / "snapshot", format=format)
    restored = load_backtest_dataset_dir(path)
    assert restored.bars.equals(dataset.bars)
    assert restored.factors.equals(dataset.factors)
    assert restored.prices.equals(dataset.prices)
    weights = pl.DataFrame({"date": ["2024-01-01"], "symbol": ["A"], "weight": [0.25]})
    weight_path = tmp_path / "weights.csv"
    weights.write_csv(weight_path)
    assert load_target_weights(weight_path)["A"].to_list() == [0.25]
