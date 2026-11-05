"""Analyzer hooks for event-driven backtests."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from ..contracts import Bar, Fill, Order, OrderRecord, OrderStatus

if TYPE_CHECKING:
    from .broker import BrokerSimulator
    from .dispatch import StrategyHooks

AnalyzerResult = dict[str, Any]


class EventAnalyzer:
    """Base class for event-driven analyzers.

    Subclasses can observe the event loop without changing strategy or broker
    semantics. Hook methods are optional; override only the events needed.
    """

    name: str | None = None
    # Conservative for existing/custom analyzers that inspect broker history.
    requires_records = True

    @property
    def result_name(self) -> str:
        return self.name or self.__class__.__name__

    def on_start(self, *, broker: BrokerSimulator, strategy: StrategyHooks) -> None:
        """Called once before the event loop starts."""

    def before_bar(self, bar: Bar, *, broker: BrokerSimulator, strategy: StrategyHooks) -> None:
        """Called before strategy logic and broker fills for each bar."""

    def on_order(
        self,
        order: Order,
        record: OrderRecord,
        *,
        broker: BrokerSimulator,
        strategy: StrategyHooks,
    ) -> None:
        """Called after a strategy order is submitted to the broker."""

    def on_fill(self, fill: Fill, *, broker: BrokerSimulator, strategy: StrategyHooks) -> None:
        """Called after the broker fills an order."""

    def after_bar(self, bar: Bar, *, broker: BrokerSimulator, strategy: StrategyHooks) -> None:
        """Called after fills and equity marking for each bar."""

    def after_slice(self, timestamp, *, broker: BrokerSimulator, strategy: StrategyHooks) -> None:
        """Called once after the whole timestamp has been marked and filled."""

    def on_finish(self, *, broker: BrokerSimulator, strategy: StrategyHooks) -> AnalyzerResult:
        """Called once after the event loop ends."""
        return {}


class OrderFlowAnalyzer(EventAnalyzer):
    """Count event-loop order and fill activity."""

    name = "order_flow"

    def __init__(self) -> None:
        self.bars = 0
        self.submitted = 0
        self.accepted = 0
        self.filled = 0
        self.partially_filled = 0
        self.rejected = 0
        self.cancelled = 0
        self.expired = 0
        self.fills = 0
        self.gross_notional = 0.0
        self.commission = 0.0
        self.slippage = 0.0
        self._seen_order_records = 0

    def before_bar(self, bar: Bar, *, broker: BrokerSimulator, strategy: StrategyHooks) -> None:
        self.bars += 1

    def on_order(
        self,
        order: Order,
        record: OrderRecord,
        *,
        broker: BrokerSimulator,
        strategy: StrategyHooks,
    ) -> None:
        self._count_new_order_records(broker)

    def _count_new_order_records(self, broker: BrokerSimulator) -> None:
        for record in broker.order_records[self._seen_order_records :]:
            self._count_status(record.status)
        self._seen_order_records = len(broker.order_records)

    def on_fill(self, fill: Fill, *, broker: BrokerSimulator, strategy: StrategyHooks) -> None:
        self.fills += 1
        self.gross_notional += abs(fill.quantity * fill.price)
        self.commission += fill.commission
        self.slippage += fill.slippage

    def after_bar(self, bar: Bar, *, broker: BrokerSimulator, strategy: StrategyHooks) -> None:
        self._count_new_order_records(broker)

    def on_finish(self, *, broker: BrokerSimulator, strategy: StrategyHooks) -> AnalyzerResult:
        self._count_new_order_records(broker)
        return {
            "bars": self.bars,
            "submitted": self.submitted,
            "accepted": self.accepted,
            "filled": self.filled,
            "partial": self.partially_filled,
            "partially_filled": self.partially_filled,
            "rejected": self.rejected,
            "cancelled": self.cancelled,
            "expired": self.expired,
            "fills": self.fills,
            "gross_notional": self.gross_notional,
            "commission": self.commission,
            "slippage": self.slippage,
        }

    def _count_status(self, status: OrderStatus) -> None:
        if status == OrderStatus.SUBMITTED:
            self.submitted += 1
        elif status == OrderStatus.ACCEPTED:
            self.accepted += 1
        elif status == OrderStatus.FILLED:
            self.filled += 1
        elif status == OrderStatus.PARTIALLY_FILLED:
            self.partially_filled += 1
        elif status == OrderStatus.REJECTED:
            self.rejected += 1
        elif status == OrderStatus.CANCELLED:
            self.cancelled += 1
        elif status == OrderStatus.EXPIRED:
            self.expired += 1


class EquityCurveAnalyzer(EventAnalyzer):
    """Collect equity snapshots only when explicitly requested."""

    name = "equity_curve"
    requires_records = False

    def __init__(self) -> None:
        self.points: list[dict[str, Any]] = []

    def after_bar(self, bar: Bar, *, broker: BrokerSimulator, strategy: StrategyHooks) -> None:
        self.points.append(broker.equity_snapshot(bar.timestamp))

    def on_finish(self, *, broker: BrokerSimulator, strategy: StrategyHooks) -> AnalyzerResult:
        return {"points": list(self.points), "count": len(self.points)}


class DrawdownAnalyzer(EventAnalyzer):
    """Track max drawdown from analyzer-side equity snapshots."""

    name = "drawdown"
    requires_records = False

    def __init__(self) -> None:
        self.peak: float | None = None
        self.max_drawdown = 0.0
        self.max_drawdown_pct = 0.0

    def after_bar(self, bar: Bar, *, broker: BrokerSimulator, strategy: StrategyHooks) -> None:
        # equity_snapshot is one dict holding the timestamp as well; "equity" is always a float.
        equity = float(cast(float, broker.equity_snapshot(bar.timestamp)["equity"]))
        if self.peak is None or equity > self.peak:
            self.peak = equity
        drawdown = equity - self.peak
        self.max_drawdown = min(self.max_drawdown, drawdown)
        if self.peak:
            self.max_drawdown_pct = min(self.max_drawdown_pct, drawdown / self.peak)

    def on_finish(self, *, broker: BrokerSimulator, strategy: StrategyHooks) -> AnalyzerResult:
        return {
            "max_drawdown": self.max_drawdown,
            "max_drawdown_pct": self.max_drawdown_pct,
            "peak": self.peak,
        }


class FillStatsAnalyzer(EventAnalyzer):
    """Summarize fill counts, turnover, and average fill prices by symbol."""

    name = "fill_stats"
    requires_records = False

    def __init__(self) -> None:
        self.fills = 0
        self.notional = 0.0
        self.by_symbol: dict[str, dict[str, float]] = {}

    def on_fill(self, fill: Fill, *, broker: BrokerSimulator, strategy: StrategyHooks) -> None:
        self.fills += 1
        notional = abs(fill.quantity * fill.price)
        self.notional += notional
        row = self.by_symbol.setdefault(fill.symbol, {"fills": 0.0, "quantity": 0.0, "notional": 0.0})
        row["fills"] += 1.0
        row["quantity"] += fill.quantity
        row["notional"] += notional

    def on_finish(self, *, broker: BrokerSimulator, strategy: StrategyHooks) -> AnalyzerResult:
        by_symbol = {}
        for symbol, row in self.by_symbol.items():
            quantity = row["quantity"]
            by_symbol[symbol] = {
                **row,
                "average_price": row["notional"] / quantity if quantity else 0.0,
            }
        return {"fills": self.fills, "notional": self.notional, "by_symbol": by_symbol}


class StreamingStatsAnalyzer(EventAnalyzer):
    """O(1)-history equity/fill statistics, sampled once per complete slice.

    Reports unannualized arithmetic return moments; no Sharpe/CAGR assumptions
    about the input calendar. Cash flows are not adjusted out of returns.
    """
    name = 'streaming_stats'
    requires_records = False

    def __init__(self) -> None:
        self.slices = self.fills = self.returns = 0
        self.commission = self.notional = 0.0
        self.initial_equity = self.last_equity = self.peak = 0.0
        self.max_drawdown = self.mean_return = self.m2_return = 0.0

    def on_start(self, *, broker, strategy) -> None:
        self.initial_equity = self.last_equity = self.peak = broker.state.equity()

    def on_fill(self, fill, *, broker, strategy) -> None:
        self.fills += 1
        self.commission += fill.commission
        self.notional += abs(fill.quantity * fill.price)

    def after_slice(self, timestamp, *, broker, strategy) -> None:
        equity = broker.state.equity()
        self.slices += 1
        self.peak = max(self.peak, equity)
        if self.peak > 0:
            self.max_drawdown = min(self.max_drawdown, equity / self.peak - 1)
        if self.last_equity > 0:
            value = equity / self.last_equity - 1
            self.returns += 1
            delta = value - self.mean_return
            self.mean_return += delta / self.returns
            self.m2_return += delta * (value - self.mean_return)
        self.last_equity = equity

    def on_finish(self, *, broker, strategy):
        return {'slices': self.slices, 'fills': self.fills, 'commission': self.commission,
                'notional': self.notional, 'final_equity': self.last_equity,
                'initial_equity': self.initial_equity, 'max_drawdown': self.max_drawdown,
                'return_count': self.returns, 'mean_return': self.mean_return,
                'return_variance': self.m2_return / self.returns if self.returns else 0.0}
