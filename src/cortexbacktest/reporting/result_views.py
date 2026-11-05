"""Report-oriented views over a completed backtest.

``BacktestResult`` is a data holder. Its reporting surface lives here, grouped
into small namespaces reached from the result:

- ``result.performance`` — equity curve, returns, drawdowns, period returns,
  performance summary, benchmark comparison;
- ``result.records`` — order/fill/open-order/position tables and on-disk export;
- ``result.exposure`` — exposure, group exposure, turnover;
- ``result.trades`` — round trips and PnL reconciliation.

``result.summary()`` and ``result.report()`` stay on the result itself as the
headline entry points.

Table accessors return Polars. Call ``.to_pandas()`` on a returned frame if a
pandas object is genuinely needed at a boundary.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import polars as pl

from .analyzers import trade_summary
from .exposure import (
    exposure_frame,
    exposure_summary,
    group_exposure_frame,
    turnover_frame,
    turnover_summary,
)
from .performance import (
    BenchmarkReport,
    PerformanceReport,
    benchmark_report,
    drawdown_table,
    performance_report,
    period_returns,
    returns_from_equity,
)
from .records import (
    fills_frame,
    open_orders_frame,
    order_records_frame,
    positions_frame,
    write_fills_csv,
    write_fills_ndjson,
    write_open_orders_csv,
    write_open_orders_ndjson,
    write_order_records_csv,
    write_order_records_ndjson,
    write_positions_csv,
    write_positions_ndjson,
)
from .trades import (
    pnl_reconciliation_frame,
    pnl_reconciliation_summary,
    trade_stats,
    trades_frame,
)


class _Namespace:
    """Base for the report namespaces bound to one result."""

    __slots__ = ("_result",)

    def __init__(self, result: Any) -> None:
        self._result = result

    def __repr__(self) -> str:
        return f"<{type(self).__name__} of {type(self._result).__name__}>"


class PerformanceNamespace(_Namespace):
    """``result.performance`` — equity, returns and performance statistics."""

    def equity_curve(self) -> pl.DataFrame:
        return self._result._equity_curve_pl.clone()

    def returns(self) -> pl.DataFrame:
        return returns_from_equity(self._result._equity_curve_pl,
                                   initial_equity=self._result.initial_equity or None)

    def drawdowns(self) -> pl.DataFrame:
        return drawdown_table(self._result._equity_curve_pl)

    def monthly_returns(self) -> pl.DataFrame:
        return period_returns(self._result._equity_curve_pl, "ME")

    def annual_returns(self) -> pl.DataFrame:
        return period_returns(self._result._equity_curve_pl, "YE")

    def report(self) -> PerformanceReport:
        return performance_report(self._result._equity_curve_pl)

    def benchmark(self, benchmark_equity, *, rolling_window: int = 20) -> BenchmarkReport:
        """Compare against a benchmark equity curve (timestamp/equity Polars frame)."""
        result = self._result
        # Collapse duplicate timestamps to the last observation per instant, so a
        # bar-level curve compares cleanly against a daily benchmark.
        equity = (
            result._equity_curve_pl.select(pl.col("timestamp"), pl.col("equity"))
            .group_by("timestamp")
            .last()
            .sort("timestamp")
        )
        return benchmark_report(equity, benchmark_equity, rolling_window=rolling_window)


class RecordsNamespace(_Namespace):
    """``result.records`` — event record tables and on-disk export."""

    def orders(self) -> pl.DataFrame:
        self._result._require_records()
        return order_records_frame(self._result.orders)

    def fills(self) -> pl.DataFrame:
        self._result._require_records()
        return fills_frame(self._result.fills)

    def open_orders(self) -> pl.DataFrame:
        self._result._require_records()
        return open_orders_frame(self._result.open_orders or [])

    def positions(self) -> pl.DataFrame:
        self._result._require_records()
        return positions_frame(self._result.positions or [])

    def write_dir(self, path: str | Path, *, format: Literal["csv", "ndjson"] = "csv") -> Path:
        """Stream core event records to a directory without building report JSON."""
        result = self._result
        result._require_records()
        output_dir = Path(path)
        output_dir.mkdir(parents=True, exist_ok=True)
        if format == "csv":
            result._equity_curve_pl.write_csv(output_dir / "equity_curve.csv")
            write_order_records_csv(result.orders, output_dir / "orders.csv")
            write_fills_csv(result.fills, output_dir / "fills.csv")
            write_open_orders_csv(result.open_orders or [], output_dir / "open_orders.csv")
            write_positions_csv(result.positions or [], output_dir / "positions.csv")
        elif format == "ndjson":
            result._equity_curve_pl.write_ndjson(output_dir / "equity_curve.ndjson")
            write_order_records_ndjson(result.orders, output_dir / "orders.ndjson")
            write_fills_ndjson(result.fills, output_dir / "fills.ndjson")
            write_open_orders_ndjson(result.open_orders or [], output_dir / "open_orders.ndjson")
            write_positions_ndjson(result.positions or [], output_dir / "positions.ndjson")
        else:
            raise ValueError("format must be 'csv' or 'ndjson'")
        manifest = result.run_manifest
        if manifest is not None:
            (output_dir / "run_manifest.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        return output_dir


class ExposureNamespace(_Namespace):
    """``result.exposure`` — exposure, group exposure and turnover."""

    def frame(self) -> pl.DataFrame:
        return exposure_frame(self._result._equity_curve_pl)

    def summary(self) -> dict[str, float]:
        return exposure_summary(self._result._equity_curve_pl)

    def by_group(self, groups, *, unknown_group: str = "unknown") -> pl.DataFrame:
        result = self._result
        result._require_records()
        return group_exposure_frame(
            result.positions or [],
            groups,
            equity=result._final_equity(),
            unknown_group=unknown_group,
        )

    def turnover(self) -> pl.DataFrame:
        result = self._result
        return turnover_frame(result.records.fills(), result._equity_curve_pl)

    def turnover_summary(self) -> dict[str, float]:
        result = self._result
        return turnover_summary(result.records.fills(), result._equity_curve_pl)


class TradesNamespace(_Namespace):
    """``result.trades`` — round trips and PnL reconciliation."""

    def frame(self) -> pl.DataFrame:
        self._result._require_records()
        return trades_frame(
            self._result.fills, self._result.corporate_action_events,
            contract_multipliers=self._result._contract_multipliers,
        )

    def summary(self) -> dict[str, float | int]:
        """Round-trip statistics."""
        result = self._result
        if not result.records_retained:
            return trade_stats(trades_frame([]))
        return trade_stats(self.frame())

    def fill_summary(self) -> dict[str, float | int]:
        """Per-fill statistics."""
        result = self._result
        if not result.records_retained:
            return trade_summary(fills_frame([]))
        return trade_summary(result.records.fills())

    def pnl_reconciliation(self) -> pl.DataFrame:
        result = self._result
        result._require_records()
        return pnl_reconciliation_frame(
            result.fills,
            result.positions or [],
            initial_equity=result.initial_equity,
            final_equity=result._final_equity(),
            contract_multipliers=result._contract_multipliers,
            corporate_actions=result.corporate_action_events,
        )

    def pnl_summary(self) -> dict[str, float | int]:
        result = self._result
        retained = result.records_retained
        return pnl_reconciliation_summary(
            result.fills if retained else [],
            (result.positions or []) if retained else [],
            initial_equity=result.initial_equity,
            final_equity=result._final_equity(),
            contract_multipliers=result._contract_multipliers,
            corporate_actions=result.corporate_action_events if retained else None,
        )


class ResultReportViews:
    """Reporting surface mixed into :class:`BacktestResult`."""

    __slots__ = ()

    # Data fields provided by BacktestResult.__init__; declared for type checkers.
    _equity_curve_pl: pl.DataFrame
    fills: list[Any]
    orders: list[Any]
    open_orders: list[Any] | None
    positions: list[Any] | None
    _contract_multipliers: dict[str, float] | None

    # Read-only accessors BacktestResult exposes as properties. Declaring them as
    # properties here (not plain attributes) keeps the override compatible.
    @property
    def fill_count(self) -> int: ...
    @property
    def submitted_order_count(self) -> int: ...
    @property
    def order_record_count(self) -> int: ...
    @property
    def initial_equity(self) -> float: ...
    @property
    def records_retained(self) -> bool: ...
    @property
    def run_manifest(self) -> dict[str, Any] | None: ...

    @property
    def performance(self) -> PerformanceNamespace:
        return PerformanceNamespace(self)

    @property
    def records(self) -> RecordsNamespace:
        return RecordsNamespace(self)

    @property
    def exposure(self) -> ExposureNamespace:
        return ExposureNamespace(self)

    @property
    def trades(self) -> TradesNamespace:
        return TradesNamespace(self)

    def summary(self) -> dict[str, float | int]:
        if self._equity_curve_pl.is_empty():
            return {
                "final_equity": 0.0,
                "total_return": 0.0,
                "cagr": 0.0,
                "sharpe": 0.0,
                "max_drawdown": 0.0,
                "fills": self.fill_count,
                "orders": self.submitted_order_count,
                "order_records": self.order_record_count,
            }
        performance = self.performance.report().summary
        return {
            "final_equity": self._final_equity(),
            "total_return": performance["total_return"],
            "cagr": performance["cagr"],
            "sharpe": performance["sharpe"],
            "max_drawdown": performance["max_drawdown"],
            "fills": self.fill_count,
            "orders": self.submitted_order_count,
            "order_records": self.order_record_count,
        }

    def report(self, *, include_records: bool = True) -> dict:
        if include_records:
            self._require_records()
        from .summary import event_report

        return event_report(self, include_records=include_records)

    def _require_records(self) -> None:
        if not self.records_retained:
            raise RuntimeError(
                "event records were not retained; run BacktestEngine with result_retention='full'"
            )

    def _final_equity(self) -> float:
        if self._equity_curve_pl.is_empty() or "equity" not in self._equity_curve_pl.columns:
            return 0.0
        value = self._equity_curve_pl.select(pl.col("equity").drop_nulls().last()).item()
        return 0.0 if value is None else float(value)
