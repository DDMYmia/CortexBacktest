"""Ordered counterfactual decomposition of research versus execution equity."""
from __future__ import annotations

from copy import deepcopy, replace
from dataclasses import dataclass
import math
from typing import TYPE_CHECKING, Any

from ..contracts import CorporateAction, FillPrice
from ..engine import BacktestEngine
from ..event.broker import BrokerSimulator
from ..event.feed import ColumnarBarFeed
from ..event.rebalance import RebalanceConfig, TargetWeightStrategy

if TYPE_CHECKING:
    from .research_execution import ResearchExecutionPipelineConfig


@dataclass(frozen=True, slots=True)
class ExecutionAttribution:
    reference_equity: float
    actual_equity: float
    stages: tuple[dict[str, Any], ...]
    diagnostics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        explained = sum(row["equity_effect"] for row in self.stages)
        return {
            "reference_equity": self.reference_equity,
            "actual_equity": self.actual_equity,
            "total_equity_gap": self.actual_equity - self.reference_equity,
            "reconciliation_residual": self.actual_equity - self.reference_equity - explained,
            "stages": [dict(row) for row in self.stages],
            "diagnostics": dict(self.diagnostics),
            "method": "ordered counterfactual equity differences; positive is a gain",
            "assumptions": [
                "Five additional independent runs; deterministic strategies and risk rules required",
                "Effects depend on stage order and include interactions and changed position sizing",
                "Target-weight model gap includes research cost assumptions and return conventions",
                "Unfilled effect includes liquidity limits and risk rejection; it is not a per-order causal estimate",
                "Recorded commission/slippage are diagnostics, not additional additive effects",
                "The zero-lag close scenario is an explanatory reference, not a deployable strategy",
            ],
        }


def attribute_execution_gap(
    feed: ColumnarBarFeed, target_weights, *, config: ResearchExecutionPipelineConfig,
    actual, research_final_equity: float, corporate_actions: tuple[CorporateAction, ...] = (),
    targets_at_open: bool = False,
) -> ExecutionAttribution:
    """Reconcile normalized research equity to currency-valued event equity.

    The decomposition is opt-in because it replays five progressively more
    constrained scenarios. The final stage uses the actual run, including fees.
    """
    cfg = config.broker
    reference = research_final_equity * cfg.initial_cash
    actual_equity = float(actual.summary()["final_equity"])
    if cfg.snapshot_frequency == "none":
        raise ValueError("attribution requires retained equity snapshots")
    if not all(math.isfinite(value) for value in (reference, actual_equity)):
        raise ValueError("attribution requires finite reference and actual equity")
    unlimited = ColumnarBarFeed.from_bars((replace(bar, volume=None) for group in feed for bar in group.bars), assume_sorted=True)

    def run(*, lag, rebalance, limited=False, slippage=0.0):
        from .research_execution import _execution_weights
        broker = BrokerSimulator(
            corporate_actions=corporate_actions,
            initial_cash=cfg.initial_cash, fill_price=cfg.fill_price if lag else FillPrice.CLOSE,
            allow_short=cfg.allow_short, commission_bps=0, slippage_bps=slippage,
            max_volume_share=cfg.max_volume_share if limited else 1.0,
            risk_rules=deepcopy(list(cfg.risk_rules)) if limited else [],
            cash_interest_model=cfg.cash_interest_model,
            short_borrow_fee_model=cfg.short_borrow_fee_model,
        )
        if targets_at_open:
            weights = target_weights if lag else _execution_weights(
                target_weights, {opening: close for close, opening in (cfg.opening_times or {}).items()})
        else:
            weights = _execution_weights(target_weights, cfg.opening_times) if lag else target_weights
        strategy = TargetWeightStrategy(weights, config=rebalance, broker=broker,
                                        signal_lag_sessions=0 if targets_at_open else lag,
                                        trade_at_open=bool(lag))
        return BacktestEngine(broker=broker, trade_on_close=cfg.trade_on_close if lag else True,
                              opening_times=cfg.opening_times if lag else None,
                              snapshot_frequency="final", result_retention="summary").run(feed if limited else unlimited, strategy)

    fractional = RebalanceConfig(allow_fractional=True)
    scenarios = (
        ("target_weight_model_gap", run(lag=0, rebalance=fractional)),
        ("execution_timing", run(lag=1, rebalance=fractional)),
        ("weight_constraints", run(lag=1, rebalance=config.rebalance)),
        ("unfilled_and_risk", run(lag=1, rebalance=config.rebalance, limited=True)),
        ("slippage", run(lag=1, rebalance=config.rebalance, limited=True, slippage=cfg.slippage_bps)),
        ("commission", actual),
    )
    rows = []
    previous = reference
    for name, result in scenarios:
        equity = float(result.summary()["final_equity"])
        if not math.isfinite(equity):
            raise ValueError(f"nonfinite attribution scenario: {name}")
        rows.append({"component": name, "equity_before": previous, "equity_after": equity,
                     "equity_effect": equity - previous, "fills": result.fill_count,
                     "open_orders": result.open_order_count})
        previous = equity
    return ExecutionAttribution(reference, actual_equity, tuple(rows), {
        "recorded_commission": actual.total_commission,
        "recorded_slippage": actual.total_slippage,
        "actual_open_orders": actual.open_order_count,
    })
