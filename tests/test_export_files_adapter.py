import polars as pl

from cortexbacktest import RebalanceConfig, ResearchExecutionPipelineConfig
from cortexbacktest.adapters import (
    build_export_dataset,
    build_research_bundle,
    normalize_bar_export,
    normalize_factor_export,
)


def _bar_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "timestamp": [
                "2024-01-01",
                "2024-01-01",
                "2024-01-02",
                "2024-01-02",
                "2024-01-03",
                "2024-01-03",
            ],
            "ticker": ["aapl", "msft", "aapl", "msft", "aapl", "msft"],
            "adjClose": [100.0, 100.0, 101.0, 99.0, 103.0, 98.0],
        }
    )


def _factor_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "analysis_timestamp": ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"],
            "ticker": ["aapl", "msft", "aapl", "msft"],
            "smart_money_score": [1.0, -1.0, 1.0, -1.0],
        }
    )


def test_normalize_exports_handles_common_aliases():
    bars = normalize_bar_export(_bar_frame())
    factors = normalize_factor_export(_factor_frame(), factor_name="smart_money_score")

    # Vendor aliases (timestamp/ticker/adjClose) map onto the standard schema.
    assert bars.columns == ["date", "symbol", "adj_close"]
    assert set(factors["symbol"].to_list()) == {"AAPL", "MSFT"}
    assert factors["factor"].to_list() == [1.0, -1.0, 1.0, -1.0]


def test_build_research_bundle_from_frames():
    bundle = build_research_bundle(
        _bar_frame(),
        _factor_frame(),
        forward_periods=1,
        factor_name="smart_money_score",
        research_mode="exploration",
    )

    # Wide matrices carry a date column plus one column per symbol.
    assert bundle.prices.shape == (3, 3)
    assert bundle.factor_values.shape == (2, 3)
    assert bundle.research_frame.height == 4


def test_build_research_bundle_from_csv(tmp_path):
    bars_path = tmp_path / "bars.csv"
    factors_path = tmp_path / "factors.csv"
    _bar_frame().write_csv(bars_path)
    _factor_frame().write_csv(factors_path)

    bundle = build_research_bundle(
        bars_path,
        factors_path,
        forward_periods=1,
        factor_name="smart_money_score",
        research_mode="exploration",
    )

    assert set(bundle.research_frame["symbol"].to_list()) == {"AAPL", "MSFT"}


def test_build_export_dataset_feeds_research_execution_pipeline():
    dataset = build_export_dataset(
        _bar_frame(), _factor_frame(), factor_name="smart_money_score", research_mode="exploration"
    )

    result = dataset.run_research_execution_pipeline(
        config=ResearchExecutionPipelineConfig(
            horizons=(1,),
            quantiles=2,
            top_quantile=0.5,
            bottom_quantile=None,
            rebalance=RebalanceConfig(allow_fractional=True),
        )
    )

    assert isinstance(result.target_weights_frame(), pl.DataFrame)
    assert result.summary()["fills"] > 0
