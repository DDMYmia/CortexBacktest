"""Core result data, independent of reporting."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import polars as pl
from ..contracts import CorporateActionEvent, Fill, Order, OrderRecord, OrderStatus, Position
from .analyzers import AnalyzerResult
from .manifest import RunManifest

EQUITY_COLUMNS = [
    "timestamp",
    "equity",
    "cash",
    "pending_cash",
    "long_exposure",
    "short_exposure",
    "gross_exposure",
    "net_exposure",
    "leverage",
]


@dataclass(frozen=True, slots=True, init=False)
class BacktestResult:
    """Result data with frozen field bindings, not deeply immutable contents.

    Fill/order lists and a supplied Polars table are retained by reference.
    The reporting layer subclasses this data holder to add views.
    """

    _equity_curve_pl: pl.DataFrame
    fills: list[Fill]
    orders: list[OrderRecord]
    open_orders: list[Order] | None = None
    positions: list[Position] | None = None
    algo_orders: list[dict[str, Any]] | None = None
    analyzer_results: dict[str, AnalyzerResult] | None = None
    _fill_count: int = 0
    _submitted_order_count: int = 0
    _order_record_count: int = 0
    _open_order_count: int = 0
    _total_commission: float = 0.0
    _total_slippage: float = 0.0
    _records_retained: bool = True
    _initial_equity: float = 0.0
    _contract_multipliers: dict[str, float] | None = None
    _run_manifest: RunManifest | None = None
    # What splits, dividends and other actions did to the account; reconciliation needs them.
    corporate_action_events: list[CorporateActionEvent] | None = None

    def __init__(
        self,
        equity_curve,
        fills: list[Fill],
        orders: list[OrderRecord],
        open_orders: list[Order] | None = None,
        positions: list[Position] | None = None,
        analyzer_results: dict[str, AnalyzerResult] | None = None,
        fill_count: int | None = None,
        submitted_order_count: int | None = None,
        order_record_count: int | None = None,
        open_order_count: int | None = None,
        total_commission: float | None = None,
        total_slippage: float | None = None,
        records_retained: bool = True,
        initial_equity: float = 0.0,
        contract_multipliers: dict[str, float] | None = None,
        run_manifest: RunManifest | None = None,
        corporate_action_events: list[CorporateActionEvent] | None = None,
        algo_orders: list[dict[str, Any]] | None = None,
    ) -> None:
        if isinstance(equity_curve, pl.DataFrame):
            native_equity = equity_curve
        elif type(equity_curve).__module__.split(".")[0] == "pandas":
            native_equity = pl.from_pandas(equity_curve)
        else:
            raise TypeError("equity_curve must be a pandas or Polars DataFrame")
        if native_equity.is_empty() and not native_equity.columns:
            native_equity = _empty_equity_curve_pl()
        object.__setattr__(self, "_equity_curve_pl", native_equity)
        object.__setattr__(self, "fills", fills)
        object.__setattr__(self, "orders", orders)
        object.__setattr__(self, "open_orders", open_orders)
        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "analyzer_results", analyzer_results)
        object.__setattr__(self, "algo_orders", algo_orders or [])
        object.__setattr__(self, "_fill_count", len(fills) if fill_count is None else fill_count)
        object.__setattr__(
            self,
            "_submitted_order_count",
            sum(1 for record in orders if record.status == OrderStatus.SUBMITTED)
            if submitted_order_count is None
            else submitted_order_count,
        )
        object.__setattr__(
            self,
            "_order_record_count",
            len(orders) if order_record_count is None else order_record_count,
        )
        object.__setattr__(
            self,
            "_open_order_count",
            len(open_orders or []) if open_order_count is None else open_order_count,
        )
        object.__setattr__(self, "_records_retained", records_retained)
        object.__setattr__(
            self,
            "_total_commission",
            sum(fill.commission for fill in fills)
            if total_commission is None
            else total_commission,
        )
        object.__setattr__(
            self,
            "_total_slippage",
            sum(fill.slippage for fill in fills) if total_slippage is None else total_slippage,
        )
        object.__setattr__(self, "_initial_equity", float(initial_equity))
        object.__setattr__(self, "_contract_multipliers", dict(contract_multipliers or {}))
        object.__setattr__(self, "_run_manifest", run_manifest)
        object.__setattr__(self, "corporate_action_events", list(corporate_action_events or []))

    @property
    def run_manifest(self) -> dict[str, Any] | None:
        """Immutable, JSON-safe provenance for this run when generated by the engine."""
        return None if self._run_manifest is None else self._run_manifest.to_dict()

    @property
    def records_retained(self) -> bool:
        return self._records_retained

    @property
    def initial_equity(self) -> float:
        return self._initial_equity

    @property
    def fill_count(self) -> int:
        return self._fill_count

    @property
    def total_commission(self) -> float:
        return self._total_commission

    @property
    def total_slippage(self) -> float:
        return self._total_slippage

    @property
    def submitted_order_count(self) -> int:
        return self._submitted_order_count

    @property
    def order_record_count(self) -> int:
        return self._order_record_count

    @property
    def open_order_count(self) -> int:
        return self._open_order_count


def _empty_equity_curve_pl() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "timestamp": pl.Datetime("us", "UTC"),
            **{column: pl.Float64 for column in EQUITY_COLUMNS if column != "timestamp"},
        }
    )
