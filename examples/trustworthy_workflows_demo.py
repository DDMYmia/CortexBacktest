"""Run the P0/P1/P2 workflows on synthetic data, without network access.

    python examples/trustworthy_workflows_demo.py --output /tmp/fusion-workflows
"""
from __future__ import annotations

import argparse
from datetime import UTC, datetime, timedelta
import json
from pathlib import Path

import polars as pl

from cortexbacktest import (
    Bar, BrokerExecutionConfig, ContractRoll, Order, RebalanceConfig,
    ResearchExecutionPipelineConfig, Side, Strategy, load_backtest_dataset,
    run_research_execution_pipeline,
)
from cortexbacktest.data.calendar import TradingCalendar
from cortexbacktest.event.financing import SettlementModel
from cortexbacktest.event.optimization import optimize_strategy
from cortexbacktest.event.timeframes import resample_bars
from cortexbacktest.pipelines import CostScenario, run_futures_roll_backtest, run_robustness_study


class PeakStrategy(Strategy):
    """Known synthetic objective for demonstrating search, not a trading signal."""
    def __init__(self, x=0):
        self.x = x
        self.done = False

    def on_bar(self, bar):
        if self.done:
            return []
        self.done = True
        return [Order(bar.symbol, bar.timestamp, Side.BUY, 101 - (self.x - 5.3)**2)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    start = datetime(2026, 1, 1, tzinfo=UTC)
    bars = [Bar("X", start + timedelta(days=i), 100+i, 100+i, 100+i, 100+i, 10000) for i in range(12)]
    grid = {"x": [i / 10 for i in range(101)]}
    exhaustive = optimize_strategy(bars, PeakStrategy, **grid)
    adaptive = optimize_strategy(bars, PeakStrategy, **grid, method="adaptive", max_tries=20,
                                 random_state=3, checkpoint_path=args.output / "adaptive.sqlite")
    robustness = run_robustness_study(bars, PeakStrategy, {"x": [2, 5, 8]},
        train_sessions=6, test_sessions=3, costs=[CostScenario("baseline"), CostScenario("costs", 1, 1)])
    robustness.write_dir(args.output / "robustness")
    market = pl.DataFrame([dict(date=bar.timestamp, symbol=symbol, open=bar.open,
                               high=bar.high, low=bar.low, close=bar.close, volume=10000)
                           for bar in bars for symbol in ["X", "Y"]])
    factors = pl.DataFrame([dict(date=bar.timestamp, symbol=symbol, factor=value)
                            for bar in bars for symbol, value in [("X", 2), ("Y", 1)]])
    pipeline = run_research_execution_pipeline(load_backtest_dataset(bars=market, factors=factors),
        config=ResearchExecutionPipelineConfig(horizons=(1,), quantiles=2,
            top_quantile=0.5, bottom_quantile=None, explain_execution=True,
            rebalance=RebalanceConfig(cash_buffer=0.02),
            broker=BrokerExecutionConfig(commission_bps=1, slippage_bps=1)))
    futures = [Bar(symbol, start + timedelta(days=i), price, price, price, price, 100)
               for i in range(6) for symbol, price in [("F1", 100+i), ("F2", 110+i)]]
    roll = run_futures_roll_backtest(futures, [ContractRoll(start + timedelta(days=2), "F1", "F2")],
                                    contract_multipliers={"F1": 10, "F2": 10}, commission_bps=1)
    calendar = TradingCalendar.from_sessions(["2026-09-04", "2026-09-08", "2026-09-09"])
    settlement = SettlementModel.trading_days(calendar, delay_days=2)
    minute_bars = [Bar("X", start + timedelta(minutes=i), 100+i, 100+i, 100+i, 100+i) for i in range(5)]
    payload = {
        "completed_5m_available_at": resample_bars(minute_bars, "5m", source_timeframe="1m")[0].timestamp.isoformat(),
        "trading_day_settlement": settlement.settlement_time(datetime(2026, 9, 4, tzinfo=UTC)).isoformat(),
        "search": {"exhaustive_cases": len(exhaustive.rows), "adaptive_cases": len(adaptive.rows),
                   "exhaustive_best": exhaustive.best_row(), "adaptive_best": adaptive.best_row()},
        "attribution": pipeline.attribution.to_dict(),
        "futures_roll": roll.report(),
    }
    (args.output / "summary.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(f"Wrote workflow reports to {args.output}")


if __name__ == "__main__":
    main()
