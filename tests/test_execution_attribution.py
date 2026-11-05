from dataclasses import replace
from datetime import datetime, timedelta

import polars as pl
import pytest

from cortexbacktest import (
    BacktestEngine, Bar, BrokerExecutionConfig, BrokerSimulator, ColumnarBarFeed,
    RebalanceConfig, ResearchExecutionPipelineConfig, TargetWeightStrategy,
    load_backtest_dataset, run_research_execution_pipeline,
)
from cortexbacktest.pipelines import attribute_execution_gap


def attribution(*, volume=10000, commission=0, slippage=0, rising=False, fractional=True):
    timestamps = [datetime(2026, 1, 1) + timedelta(days=i) for i in range(4)]
    feed = ColumnarBarFeed.from_bars([
        Bar("X", ts, 100+i*10 if rising else 100, 100+i*10 if rising else 100,
            100+i*10 if rising else 100, 100+i*10 if rising else 100, volume)
        for i, ts in enumerate(timestamps)])
    targets = pl.DataFrame({"date": timestamps, "X": [0.5]*4})
    cfg = ResearchExecutionPipelineConfig(rebalance=RebalanceConfig(allow_fractional=fractional),
        broker=BrokerExecutionConfig(initial_cash=1000, commission_bps=commission, slippage_bps=slippage))
    broker = BrokerSimulator(initial_cash=1000, commission_bps=commission, slippage_bps=slippage,
                             fill_price=cfg.broker.fill_price)
    result = BacktestEngine(broker=broker, trade_on_close=False).run(feed,
        TargetWeightStrategy(targets, broker=broker, config=cfg.rebalance, signal_lag_sessions=1))
    return attribute_execution_gap(feed, targets, config=cfg, actual=result, research_final_equity=1)


def test_flat_frictionless_execution_has_zero_gap():
    result = attribution().to_dict()
    assert result["total_equity_gap"] == 0
    assert all(row["equity_effect"] == 0 for row in result["stages"])


def test_commission_and_slippage_effects_reconcile_without_double_counting():
    result = attribution(commission=10, slippage=10).to_dict()
    effects = {row["component"]: row["equity_effect"] for row in result["stages"]}
    assert effects["commission"] < 0
    assert effects["slippage"] < 0
    assert result["diagnostics"]["recorded_commission"] > 0
    assert sum(effects.values()) == pytest.approx(result["total_equity_gap"])
    assert result["reconciliation_residual"] == pytest.approx(0, abs=1e-10)


def test_no_volume_explains_missed_rising_market_profit():
    result = attribution(volume=0, rising=True).to_dict()
    effects = {row["component"]: row["equity_effect"] for row in result["stages"]}
    assert effects["execution_timing"] < 0
    assert effects["unfilled_and_risk"] < 0
    assert result["diagnostics"]["actual_open_orders"] > 0
    assert result["actual_equity"] == 1000
    assert result["reconciliation_residual"] == pytest.approx(0, abs=1e-10)


def test_pipeline_attribution_is_opt_in_and_exported():
    dates = [datetime(2026, 1, 1) + timedelta(days=i) for i in range(6)]
    bars = pl.DataFrame([dict(date=day, symbol=symbol, open=100+i, high=100+i,
                             low=100+i, close=100+i, volume=10000)
                         for i, day in enumerate(dates) for symbol in ["X", "Y"]])
    factors = pl.DataFrame([dict(date=day, symbol=symbol, factor=value)
                           for day in dates for symbol, value in [("X", 1), ("Y", 2)]])
    dataset = load_backtest_dataset(bars=bars, factors=factors)
    cfg = ResearchExecutionPipelineConfig(horizons=(1,), quantiles=2, bottom_quantile=None,
        top_quantile=0.5, rebalance=RebalanceConfig(cash_buffer=0.01), explain_execution=True)
    explained = run_research_execution_pipeline(dataset, config=cfg)
    plain = run_research_execution_pipeline(dataset, config=replace(cfg, explain_execution=False))
    assert plain.attribution is None
    assert explained.event.summary() == plain.event.summary()
    report = explained.report(include_records=False)["execution_attribution"]
    assert len(report["stages"]) == 6
    assert report["reconciliation_residual"] == pytest.approx(0, abs=1e-8)
