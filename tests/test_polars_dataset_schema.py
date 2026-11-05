from __future__ import annotations

import polars as pl
import pytest

from cortexbacktest import (
    Bar,
    Order,
    Side,
    Strategy,
    load_backtest_dataset_dir,
    load_backtest_dataset,
    save_backtest_dataset,
)
from cortexbacktest.datasets import BacktestDataset, bars_to_events, normalize_bars, validate_bars


def _bars_pl() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "date": [
                "2024-01-02",
                "2024-01-01",
                "2024-01-01",
                "2024-01-02",
            ],
            "symbol": ["b", "a", "b", "a"],
            "adj_close": [19.0, 10.0, 20.0, 11.0],
        }
    )


def _factors_pl() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"],
            "symbol": ["A", "B", "A", "B"],
            "factor": [1.0, -1.0, 1.0, -1.0],
        }
    )


def test_normalize_bars_pl_fills_ohlc_and_sorts() -> None:
    bars = normalize_bars(_bars_pl())

    assert isinstance(bars, pl.DataFrame)
    assert bars.columns == ["date", "symbol", "open", "high", "low", "close", "volume", "adj_close"]
    assert bars["symbol"].to_list() == ["A", "B", "A", "B"]
    assert bars["open"].to_list() == [10.0, 20.0, 11.0, 19.0]
    assert validate_bars(bars).ok


def test_validate_bars_pl_rejects_duplicates_and_bad_prices() -> None:
    bars = normalize_bars(_bars_pl())
    bad = pl.concat(
        [
            bars.with_columns(pl.when(pl.arange(0, pl.len()) == 0).then(0.0).otherwise(pl.col("close")).alias("close")),
            bars.head(1),
        ]
    )

    report = validate_bars(bad)

    assert not report.ok
    assert any("duplicate date/symbol" in error for error in report.errors)
    assert any("close contains" in error for error in report.errors)


def test_bars_to_events_pl_returns_sorted_bar_contracts_with_vwap() -> None:
    source = _bars_pl().with_columns(pl.Series("vwap", [19.2, 10.2, 20.2, 11.2]))

    bars = bars_to_events(source)

    assert [bar.symbol for bar in bars[:2]] == ["A", "B"]
    assert isinstance(bars[0], Bar)
    assert bars[0].close == 10.0
    assert bars[0].vwap == 10.2


def test_bars_to_events_pl_rejects_bad_bar_data() -> None:
    source = _bars_pl().with_columns(pl.when(pl.col("symbol") == "a").then(-1.0).otherwise(20.0).alias("adj_close"))

    with pytest.raises(ValueError, match="adj_close contains"):
        bars_to_events(source)


class DatasetBuyOnceStrategy(Strategy):
    def __init__(self) -> None:
        self.bought = False

    def on_bars(self, bars: list[Bar]) -> list[Order]:
        if self.bought:
            return []
        self.bought = True
        return [Order(symbol=bar.symbol, timestamp=bar.timestamp, side=Side.BUY, quantity=1) for bar in bars[:1]]


def test_load_backtest_dataset_pl_returns_polars_tables_and_runs_events() -> None:
    dataset = load_backtest_dataset(bars=_bars_pl(), factors=_factors_pl())

    assert isinstance(dataset, BacktestDataset)
    assert isinstance(dataset.bars, pl.DataFrame)
    assert isinstance(dataset.prices, pl.DataFrame)
    assert isinstance(dataset.factor_values, pl.DataFrame)
    assert isinstance(dataset.returns, pl.DataFrame)
    assert dataset.prices.columns == ["date", "A", "B"]
    assert dataset.factor_values.columns == ["date", "A", "B"]
    assert dataset.quality.ok

    result = dataset.run_event_backtest(DatasetBuyOnceStrategy())

    assert len(result.fills) == 1
    assert result.fills[0].symbol == "A"


def test_backtest_dataset_pl_runs_factor_research_with_polars_outputs() -> None:
    dataset = load_backtest_dataset(bars=_bars_pl(), factors=_factors_pl())

    report = dataset.run_factor_research(horizons=(1,), quantiles=2, top_quantile=0.5, bottom_quantile=0.5)

    assert isinstance(report.summary(), pl.DataFrame)
    assert isinstance(report.coverage(), pl.DataFrame)
    assert isinstance(report.turnover(), pl.DataFrame)
    assert isinstance(report.target_weights_frame(), pl.DataFrame)
    assert isinstance(report.equity_curve_frame(), pl.DataFrame)
    assert report.summary()["horizon"].to_list() == [1]


def test_save_and_load_backtest_dataset_pl_directory_round_trips_csv(tmp_path) -> None:
    dataset = load_backtest_dataset(bars=_bars_pl(), factors=_factors_pl())
    output_dir = tmp_path / "dataset_pl"

    saved = save_backtest_dataset(dataset, output_dir, format="csv")
    loaded = load_backtest_dataset_dir(saved)

    assert (saved / "manifest.json").exists()
    assert isinstance(loaded, BacktestDataset)
    assert loaded.bars.shape == dataset.bars.shape
    assert loaded.factors.shape == dataset.factors.shape
    assert loaded.prices.columns == dataset.prices.columns
    assert loaded.factor_values.columns == dataset.factor_values.columns


def test_backtest_dataset_pl_save_method_round_trips(tmp_path) -> None:
    dataset = load_backtest_dataset(bars=_bars_pl(), factors=_factors_pl())

    loaded = load_backtest_dataset_dir(dataset.save(tmp_path / "dataset_pl"))

    assert loaded.prices.equals(dataset.prices)
    assert loaded.factor_values.equals(dataset.factor_values)
