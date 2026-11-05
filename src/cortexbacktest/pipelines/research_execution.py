"""Research-to-execution pipeline.

This layer keeps the stable orchestration API separate from the research,
dataset, and event-engine internals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from copy import deepcopy
from datetime import datetime
from math import isfinite
from typing import TYPE_CHECKING, Any, Literal

import polars as pl

from ..contracts import FillPrice
from ..event import RebalanceConfig, TargetWeightStrategy
from ..event.broker import BrokerSimulator
from ..event.engine import BacktestEngine, BacktestResult
from ..event.financing import CashInterestModel, ShortBorrowFeeModel
from ..event.feed import ColumnarBarFeed
from ..datasets.schema import normalize_bars, validate_bars
from ..reporting.reports import research_report
from ..reporting.summary import frame_records
from .attribution import ExecutionAttribution, attribute_execution_gap

if TYPE_CHECKING:
    from ..datasets import BacktestDataset
    from ..research import FactorPreprocessConfig, FactorResearchReport


SnapshotFrequency = Literal["bar", "session", "final", "none"]


@dataclass(frozen=True, slots=True)
class BrokerExecutionConfig:
    """Broker and event-loop settings for pipeline execution.

    ``opening_times`` maps each close-stamped bar to its aware opening instant.
    This mode matches only opening prints, keeps close-time valuation, and
    retains session volume as a capacity approximation, not auction liquidity.
    """

    initial_cash: float = 100_000.0
    commission_bps: float = 0.0
    slippage_bps: float = 0.0
    fill_price: FillPrice = FillPrice.OPEN
    allow_short: bool = True
    max_volume_share: float = 1.0
    trade_on_close: bool = False
    snapshot_frequency: SnapshotFrequency = "bar"
    risk_rules: tuple[Any, ...] = field(default_factory=tuple)
    opening_times: dict[datetime, datetime] | None = None
    cash_interest_model: CashInterestModel | None = None
    short_borrow_fee_model: ShortBorrowFeeModel | None = None


@dataclass(frozen=True, slots=True)
class ResearchExecutionPipelineConfig:
    """Configuration for factor research followed by event execution."""

    horizons: tuple[int, ...] = (1, 5, 20)
    quantiles: int = 5
    top_quantile: float = 0.2
    bottom_quantile: float | None = 0.2
    transaction_cost_bps: float = 0.0
    preprocessing: FactorPreprocessConfig | None = None
    rebalance: RebalanceConfig = field(default_factory=RebalanceConfig)
    broker: BrokerExecutionConfig = field(default_factory=BrokerExecutionConfig)
    explain_execution: bool = False


@dataclass(frozen=True, slots=True)
class ResearchExecutionPipelineResult:
    """Result bundle for the research-to-execution workflow."""

    research: FactorResearchReport
    event: BacktestResult
    target_weights: pl.DataFrame
    config: ResearchExecutionPipelineConfig
    attribution: ExecutionAttribution | None = None

    def target_weights_frame(self) -> pl.DataFrame:
        return self.target_weights

    def summary(self) -> dict[str, Any]:
        event_summary = self.event.summary()
        primary_horizon = self.config.horizons[0]
        primary_row = _research_summary_row(self.research, primary_horizon)
        return {
            "primary_horizon": primary_horizon,
            "research_mean_ic": _float_or_none(primary_row.get("mean_ic")),
            "research_mean_rank_ic": _float_or_none(primary_row.get("mean_rank_ic")),
            "research_mean_spread": _float_or_none(primary_row.get("mean_spread")),
            "research_final_equity": _float_or_none(self.research.final_equity),
            "event_final_equity": _float_or_none(event_summary.get("final_equity")),
            "event_total_return": _float_or_none(event_summary.get("total_return")),
            "event_sharpe": _float_or_none(event_summary.get("sharpe")),
            "event_max_drawdown": _float_or_none(event_summary.get("max_drawdown")),
            "fills": event_summary.get("fills", 0),
            "orders": event_summary.get("orders", 0),
        }

    def report(self, *, include_records: bool = True) -> dict[str, Any]:
        report = {
            "summary": self.summary(),
            "research": research_report(self.research, include_records=include_records),
            "event": self.event.report(include_records=include_records),
        }
        if include_records:
            report["target_weights"] = frame_records(self.target_weights_frame())
        if self.attribution is not None:
            report["execution_attribution"] = self.attribution.to_dict()
        return report


def run_research_execution_pipeline(
    dataset: BacktestDataset,
    *,
    config: ResearchExecutionPipelineConfig | None = None,
    run_metadata: dict[str, Any] | None = None,
    execution_bars: pl.DataFrame | None = None,
) -> ResearchExecutionPipelineResult:
    """Run factor research, convert targets to orders, and execute through bars."""
    cfg = config or ResearchExecutionPipelineConfig()
    event_feed = _execution_feed(dataset, execution_bars)
    research = dataset.run_factor_research(
        preprocessing=cfg.preprocessing,
        horizons=cfg.horizons,
        quantiles=cfg.quantiles,
        top_quantile=cfg.top_quantile,
        bottom_quantile=cfg.bottom_quantile,
        transaction_cost_bps=cfg.transaction_cost_bps,
    )
    return _execute_research(research, event_feed, dataset=dataset, cfg=cfg, run_metadata=run_metadata)


def _execution_feed(dataset, execution_bars, *, start_after=None) -> ColumnarBarFeed:
    if execution_bars is None:
        if dataset.corporate_actions:
            raise ValueError("Corporate actions require explicit unadjusted execution_bars")
        raw = dataset.bars
    else:
        required = {"date", "symbol", "open", "high", "low", "close", "volume"}
        if not required.issubset(execution_bars.columns) or "adj_close" in execution_bars.columns:
            raise ValueError("execution_bars must supply raw OHLCV without adj_close")
        raw = normalize_bars(execution_bars)
        validate_bars(raw, calendar=dataset.calendar, resolution=dataset.resolution).raise_if_failed()
        if not raw.select("date", "symbol").equals(dataset.bars.select("date", "symbol")):
            raise ValueError("Execution bar identities and dates must match the research dataset")
        # Normalization supplied a temporary close copy for validation only;
        # the execution feed explicitly uses the caller's raw OHLC values.
        raw = raw.drop("adj_close")
    if start_after is not None:
        raw = raw.filter(pl.col("date") > start_after)
    return ColumnarBarFeed.from_polars(raw)


def _execute_research(research, event_feed, *, dataset, cfg, run_metadata=None, targets_at_open=False) -> ResearchExecutionPipelineResult:
    if cfg.explain_execution and cfg.broker.snapshot_frequency == "none":
        raise ValueError("attribution requires retained equity snapshots")
    target_weights = research.target_weights_frame()
    event = _execute_targets(target_weights, event_feed, dataset=dataset, cfg=cfg,
                             run_metadata=run_metadata, targets_at_open=targets_at_open)
    attribution = None
    if cfg.explain_execution:
        from ..event.feed import market_slices_from_bars
        reference = _float_or_none(research.final_equity)
        if reference is None:
            raise ValueError("attribution requires nonempty research equity")
        attribution = attribute_execution_gap(market_slices_from_bars(event_feed), target_weights,
                                             config=cfg, actual=event, research_final_equity=reference,
                                             corporate_actions=dataset.corporate_actions, targets_at_open=targets_at_open)
    return ResearchExecutionPipelineResult(
        research=research,
        event=event,
        target_weights=target_weights,
        config=cfg,
        attribution=attribution,
    )


def _execute_targets(target_weights, event_feed, *, dataset, cfg, run_metadata=None,
                     targets_at_open=False, decision_times=False):
    broker = BrokerSimulator(
        initial_cash=cfg.broker.initial_cash,
        corporate_actions=dataset.corporate_actions,
        commission_bps=cfg.broker.commission_bps,
        slippage_bps=cfg.broker.slippage_bps,
        fill_price=cfg.broker.fill_price,
        allow_short=cfg.broker.allow_short,
        max_volume_share=cfg.broker.max_volume_share,
        risk_rules=deepcopy(list(cfg.broker.risk_rules)),
        cash_interest_model=cfg.broker.cash_interest_model,
        short_borrow_fee_model=cfg.broker.short_borrow_fee_model,
    )
    strategy = TargetWeightStrategy(
        target_weights if targets_at_open or decision_times else _execution_weights(target_weights, cfg.broker.opening_times),
        config=cfg.rebalance,
        broker=broker,
        signal_lag_sessions=0 if targets_at_open or decision_times else 1,
        decision_times=decision_times,
        trade_at_open=True,
    )
    engine = BacktestEngine(
        broker=broker,
        trade_on_close=cfg.broker.trade_on_close,
        snapshot_frequency=cfg.broker.snapshot_frequency,
        build_manifest=run_metadata is not None,
        run_metadata=run_metadata,
        opening_times=cfg.broker.opening_times,
    )
    return engine.run(event_feed, strategy)


def _execution_weights(target_weights, opening_times):
    """Keep session alignment when a close label and its open use different days."""
    if opening_times is None:
        return target_weights
    return target_weights.with_columns(pl.col("date").replace_strict(opening_times))


def _research_summary_row(research, horizon: int) -> dict[str, Any]:
    rows = research.summary().filter(pl.col("horizon") == horizon)
    return rows.row(0, named=True) if not rows.is_empty() else {}


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    result = float(value)
    return result if isfinite(result) else None
