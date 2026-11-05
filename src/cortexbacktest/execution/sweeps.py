"""Execute research sweep target weights through the event engine."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from typing import TYPE_CHECKING, Callable

import polars as pl

from ..contracts import FillPrice
from ..event.feed import ColumnarBarFeed
from ..event.parallel import ParallelBackend, ProcessPlan, run_cases, validate_backend, worker_feed
from ..scheduling import SweepScheduling
from ..event import BacktestEngine, BrokerSimulator, RebalanceConfig, TargetWeightStrategy
from ..event.costs import CommissionModel, SlippageModel
from ..event.risk import RiskRule
from ..research.sweeps import FactorSweepResult, best_case_in

if TYPE_CHECKING:
    from ..event.engine import BacktestResult


BrokerFactory = Callable[[], BrokerSimulator]

@dataclass(frozen=True, slots=True)
class ExecutionSweepResult:
    """Event execution results for each factor sweep case."""

    factor_sweep: FactorSweepResult
    event_results: "dict[str, BacktestResult]"
    execution: dict = field(default_factory=dict)

    def summary(self) -> pl.DataFrame:
        rows = []
        research_summary = self.factor_sweep.stability()
        for case, result in self.event_results.items():
            event_summary = result.summary()
            research_rows = research_summary.filter(pl.col("case") == case)
            research_row = research_rows.row(0, named=True) if not research_rows.is_empty() else {}
            rows.append(
                {
                    "case": case,
                    "research_final_equity": research_row.get("final_equity"),
                    "research_avg_rank_ic": research_row.get("avg_rank_ic"),
                    "event_final_equity": event_summary["final_equity"],
                    "event_total_return": event_summary["total_return"],
                    "event_sharpe": event_summary["sharpe"],
                    "event_max_drawdown": event_summary["max_drawdown"],
                    "orders": event_summary["orders"],
                    "fills": event_summary["fills"],
                    "order_records": event_summary["order_records"],
                    "commission": result.total_commission,
                    "slippage": result.total_slippage,
                }
            )
        if not rows:
            return pl.DataFrame(
                {
                    "case": [],
                    "research_final_equity": [],
                    "research_avg_rank_ic": [],
                    "event_final_equity": [],
                    "event_total_return": [],
                    "event_sharpe": [],
                    "event_max_drawdown": [],
                    "orders": [],
                    "fills": [],
                    "order_records": [],
                    "commission": [],
                    "slippage": [],
                }
            )
        return pl.DataFrame(rows).sort(
            ["event_final_equity", "research_avg_rank_ic"],
            descending=[True, True],
        )

    def best_case(self, *, metric: str = "event_final_equity") -> str | None:
        return best_case_in(self.summary(), metric)


def run_execution_sweep(
    factor_sweep: FactorSweepResult,
    bars,
    *,
    broker_factory: BrokerFactory | None = None,
    initial_cash: float = 100_000.0,
    commission_bps: float = 0.0,
    slippage_bps: float = 0.0,
    commission_model: CommissionModel | None = None,
    slippage_model: SlippageModel | None = None,
    fill_price: FillPrice = FillPrice.OPEN,
    allow_short: bool = True,
    max_volume_share: float = 1.0,
    risk_rules: list[RiskRule] | None = None,
    rebalance_config: RebalanceConfig | None = None,
    trade_on_close: bool = False,
    max_workers: int | None = 1,
    parallel_backend: ParallelBackend = "auto",
    scheduling: SweepScheduling | None = None,
) -> ExecutionSweepResult:
    """Execute target weights using one reusable feed and bounded worker tasks.

    Auto retains a real pilot result and chooses process execution only when
    estimated saved time exceeds the scheduling policy's startup budget.
    """
    if max_workers is not None and max_workers < 1:
        raise ValueError("max_workers must be positive or None")
    validate_backend(parallel_backend)
    custom = broker_factory is not None or commission_model is not None or slippage_model is not None
    if max_workers != 1 and parallel_backend == "process" and custom:
        raise ValueError("process execution sweeps require default broker construction; use parallel_backend='thread' for custom broker_factory or cost models")
    workers = max_workers or os.cpu_count() or 1
    feed = bars if isinstance(bars, ColumnarBarFeed) else ColumnarBarFeed.from_bars(bars)
    rebalance = rebalance_config or RebalanceConfig()
    count = len(factor_sweep.reports)
    def run_one(item):
        case, report = item
        return case, _run_execution_sweep_case(
            report=report, bars=feed, broker_factory=broker_factory,
            initial_cash=initial_cash, commission_bps=commission_bps, slippage_bps=slippage_bps,
            commission_model=commission_model, slippage_model=slippage_model, fill_price=fill_price,
            allow_short=allow_short, max_volume_share=max_volume_share, risk_rules=risk_rules,
            rebalance_config=rebalance, trade_on_close=trade_on_close)
    # Only target weights cross the process boundary, not full factor reports.
    process = None if custom else ProcessPlan(
        task=lambda item: (item[0], item[1].target_weights, initial_cash, commission_bps, slippage_bps,
                           fill_price, allow_short, max_volume_share, risk_rules, rebalance, trade_on_close),
        run=_run_execution_sweep_case_process)
    timing = {"requested_backend": parallel_backend, "workers": workers}
    event_results = dict(run_cases(
        factor_sweep.reports.items(), run_one, feed=feed,
        backend="serial" if workers == 1 or count <= 1 else parallel_backend, workers=workers,
        remaining=count - 1, policy=scheduling or SweepScheduling(), execution=timing, process=process))
    return ExecutionSweepResult(factor_sweep, event_results, timing)


def _run_execution_sweep_case(
    *,
    report,
    bars: list | ColumnarBarFeed,
    broker_factory: BrokerFactory | None,
    initial_cash: float,
    commission_bps: float,
    slippage_bps: float,
    commission_model: CommissionModel | None,
    slippage_model: SlippageModel | None,
    fill_price: FillPrice,
    allow_short: bool,
    max_volume_share: float,
    risk_rules: list[RiskRule] | None,
    rebalance_config: RebalanceConfig,
    trade_on_close: bool,
):
    broker = broker_factory() if broker_factory is not None else BrokerSimulator(
        initial_cash=initial_cash,
        commission_bps=commission_bps,
        slippage_bps=slippage_bps,
        commission_model=commission_model,
        slippage_model=slippage_model,
        fill_price=fill_price,
        allow_short=allow_short,
        max_volume_share=max_volume_share,
        risk_rules=risk_rules,
    )
    strategy = TargetWeightStrategy(
        report if isinstance(report, pl.DataFrame) else report.target_weights,
        broker=broker,
        config=rebalance_config,
    )
    return BacktestEngine(
        broker=broker,
        trade_on_close=trade_on_close,
        snapshot_frequency="final",
        result_retention="summary",
        build_manifest=False,
    ).run(
        bars,
        strategy,
        assume_sorted=True,
    )


def _run_execution_sweep_case_process(args):
    (
        case,
        report,
        initial_cash,
        commission_bps,
        slippage_bps,
        fill_price,
        allow_short,
        max_volume_share,
        risk_rules,
        rebalance_config,
        trade_on_close,
    ) = args
    bars = worker_feed()
    result = _run_execution_sweep_case(
        report=report,
        bars=bars,
        broker_factory=None,
        initial_cash=initial_cash,
        commission_bps=commission_bps,
        slippage_bps=slippage_bps,
        commission_model=None,
        slippage_model=None,
        fill_price=fill_price,
        allow_short=allow_short,
        max_volume_share=max_volume_share,
        risk_rules=risk_rules,
        rebalance_config=rebalance_config,
        trade_on_close=trade_on_close,
    )
    return case, result

