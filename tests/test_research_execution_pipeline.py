import pandas as pd
import polars as pl

from cortexbacktest import (
    BrokerExecutionConfig,
    RebalanceConfig,
    ResearchExecutionPipelineConfig,
    load_backtest_dataset,
    run_research_execution_pipeline,
)


def test_pipeline_defaults_to_next_bar_open_execution():
    config = BrokerExecutionConfig()

    assert config.trade_on_close is False
    assert config.fill_price.value == "open"


def _bars() -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=6)
    rows = []
    values = {
        "A": [100.0, 102.0, 104.0, 106.0, 108.0, 110.0],
        "B": [100.0, 99.0, 98.0, 97.0, 96.0, 95.0],
        "C": [100.0, 101.0, 101.5, 102.0, 103.0, 104.0],
    }
    for date_index, date in enumerate(dates):
        for symbol, prices in values.items():
            rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "open": prices[date_index],
                    "high": prices[date_index],
                    "low": prices[date_index],
                    "close": prices[date_index],
                    "adj_close": prices[date_index],
                    "volume": 10_000,
                }
            )
    return pd.DataFrame(rows)


def _factors() -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=5)
    rows = []
    for date in dates:
        rows.extend(
            [
                {"date": date, "symbol": "A", "factor": 3.0},
                {"date": date, "symbol": "B", "factor": 1.0},
                {"date": date, "symbol": "C", "factor": 2.0},
            ]
        )
    return pd.DataFrame(rows)


def test_research_execution_pipeline_runs_full_closed_loop():
    dataset = load_backtest_dataset(bars=_bars(), factors=_factors())
    config = ResearchExecutionPipelineConfig(
        horizons=(1,),
        quantiles=3,
        top_quantile=1 / 3,
        bottom_quantile=None,
        rebalance=RebalanceConfig(cash_buffer=0.01, min_trade_value=100.0, allow_fractional=True),
        broker=BrokerExecutionConfig(initial_cash=100_000, commission_bps=1.0, slippage_bps=1.0),
    )

    result = run_research_execution_pipeline(dataset, config=config)
    summary = result.summary()
    report = result.report(include_records=False)

    assert result.research.summary().filter(pl.col("horizon") == 1)["mean_rank_ic"].item() > 0
    assert result.target_weights["A"].gt(0).any()
    assert len(result.event.fills) > 0
    assert summary["primary_horizon"] == 1
    assert summary["event_final_equity"] > 0
    assert report["summary"]["fills"] == len(result.event.fills)
    assert set(report) == {"summary", "research", "event"}


def test_dataset_exposes_research_execution_pipeline_shortcut():
    dataset = load_backtest_dataset(bars=_bars(), factors=_factors())

    result = dataset.run_research_execution_pipeline(
        config=ResearchExecutionPipelineConfig(
            horizons=(1,),
            quantiles=3,
            top_quantile=1 / 3,
            bottom_quantile=None,
            rebalance=RebalanceConfig(allow_fractional=True),
        )
    )

    assert result.summary()["fills"] > 0


def test_research_execution_pipeline_accepts_polars_dataset():
    dataset = load_backtest_dataset(
        bars=pl.from_pandas(_bars()), factors=pl.from_pandas(_factors())
    )
    config = ResearchExecutionPipelineConfig(
        horizons=(1,),
        quantiles=3,
        top_quantile=1 / 3,
        bottom_quantile=None,
        rebalance=RebalanceConfig(cash_buffer=0.01, min_trade_value=100.0, allow_fractional=True),
        broker=BrokerExecutionConfig(initial_cash=100_000, commission_bps=1.0, slippage_bps=1.0),
    )

    result = run_research_execution_pipeline(dataset, config=config)
    report = result.report(include_records=True)

    assert isinstance(result.target_weights_frame(), pl.DataFrame)
    assert isinstance(result.target_weights, pl.DataFrame)
    assert result.research.summary().filter(pl.col("horizon") == 1)["mean_rank_ic"].item() > 0
    assert result.target_weights_frame()["A"].max() > 0
    assert len(result.event.fills) > 0
    assert report["target_weights"]


def test_polars_dataset_exposes_research_execution_pipeline_shortcut():
    dataset = load_backtest_dataset(
        bars=pl.from_pandas(_bars()), factors=pl.from_pandas(_factors())
    )

    result = dataset.run_research_execution_pipeline(
        config=ResearchExecutionPipelineConfig(
            horizons=(1,),
            quantiles=3,
            top_quantile=1 / 3,
            bottom_quantile=None,
            rebalance=RebalanceConfig(allow_fractional=True),
        )
    )

    assert isinstance(result.target_weights_frame(), pl.DataFrame)
    assert result.summary()["fills"] > 0


def test_pipeline_keeps_corporate_actions_in_execution_and_attribution():
    from datetime import date
    import pytest
    from cortexbacktest import CorporateAction, CorporateActionType

    actions = [
        CorporateAction("A", date(2024, 1, 5), CorporateActionType.CASH_DIVIDEND, amount=5.0)
    ]
    dataset = load_backtest_dataset(bars=_bars(), factors=_factors(), corporate_actions=actions)
    config = ResearchExecutionPipelineConfig(
        horizons=(1,),
        quantiles=3,
        top_quantile=1 / 3,
        bottom_quantile=None,
        explain_execution=True,
        rebalance=RebalanceConfig(allow_fractional=True),
        broker=BrokerExecutionConfig(initial_cash=100000),
    )
    metadata = {
        "input_files": {"dataset": "a" * 64},
        "snapshot_as_of": "2024-02-01T00:00:00Z",
        "availability_policy": "exploration",
    }
    with pytest.raises(ValueError, match="unadjusted execution_bars"):
        dataset.run_research_execution_pipeline(config=config, run_metadata=metadata)
    result = dataset.run_research_execution_pipeline(
        config=config,
        run_metadata=metadata,
        execution_bars=pl.from_pandas(_bars().drop(columns="adj_close")),
    )
    baseline = run_research_execution_pipeline(
        load_backtest_dataset(bars=_bars(), factors=_factors()), config=config
    )
    assert result.event.summary()["final_equity"] > baseline.event.summary()["final_equity"]
    assert result.event.run_manifest["metadata"] == metadata
    assert result.event.run_manifest["broker"]["corporate_actions"]["count"] == 1
    assert result.attribution.stages[-1]["component"] == "commission"
    assert result.attribution.stages[-1]["equity_effect"] == pytest.approx(0.0)


def test_pipeline_rejects_adjusted_or_inconsistent_execution_input():
    import pytest

    data = load_backtest_dataset(bars=_bars(), factors=_factors())
    with pytest.raises(ValueError, match="raw OHLCV"):
        data.run_research_execution_pipeline(execution_bars=pl.from_pandas(_bars()))
    raw = pl.from_pandas(_bars().drop(columns="adj_close"))
    with pytest.raises(ValueError, match="identities and dates"):
        data.run_research_execution_pipeline(execution_bars=raw.head(1))
    for invalid in (float("inf"), float("nan")):
        with pytest.raises(ValueError, match="nonfinite"):
            data.run_research_execution_pipeline(
                execution_bars=raw.with_columns(pl.lit(invalid).alias("close"))
            )
