"""Round-trip trade matching from fill records."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime

import polars as pl

from ..contracts import QUANTITY_TOLERANCE, CorporateActionEvent, CorporateActionType, Fill, Position


@dataclass(slots=True)
class _Lot:
    timestamp: datetime
    side: str
    quantity: float
    price: float
    commission_per_unit: float


def trades_frame(
    fills: list[Fill],
    corporate_actions: list[CorporateActionEvent] | None = None,
    *,
    contract_multipliers: dict[str, float] | None = None,
) -> pl.DataFrame:
    """Match FIFO trades, scaling price PnL and basis but not cash commissions."""
    rows, _ = _trade_rows_and_open_lots(
        _fill_records(fills), corporate_actions=corporate_actions,
        contract_multipliers=contract_multipliers,
    )
    if not rows:
        return pl.DataFrame({column: [] for column in TRADE_COLUMNS})
    return pl.DataFrame(rows).select(TRADE_COLUMNS)


def pnl_reconciliation_frame(
    fills: list[Fill],
    positions: list[Position] | None = None,
    *,
    initial_equity: float = 0.0,
    final_equity: float | None = None,
    contract_multipliers: dict[str, float] | None = None,
    corporate_actions: list[CorporateActionEvent] | None = None,
) -> pl.DataFrame:
    """Reconcile fill-level realized/open PnL to final equity."""
    summary = pnl_reconciliation_summary(
        fills,
        positions,
        initial_equity=initial_equity,
        final_equity=final_equity,
        contract_multipliers=contract_multipliers,
        corporate_actions=corporate_actions,
    )
    return pl.DataFrame([summary]).select(PNL_RECONCILIATION_COLUMNS)


def pnl_reconciliation_summary(
    fills: list[Fill],
    positions: list[Position] | None = None,
    *,
    initial_equity: float = 0.0,
    final_equity: float | None = None,
    contract_multipliers: dict[str, float] | None = None,
    corporate_actions: list[CorporateActionEvent] | None = None,
) -> dict[str, float | int]:
    """Return scalar PnL reconciliation metrics from fill records.

    Corporate actions the account went through are replayed against the open lots in time
    order: a split rescales them, a delisting or cash merger closes them at the cash paid, a
    symbol change or stock merger carries them to the new symbol, and cash dividends are
    income of their own. Without them a split re-pairs lots at the wrong prices and the
    reconciliation error is the corporate actions themselves.
    """
    multipliers = {symbol.upper(): float(value) for symbol, value in (contract_multipliers or {}).items()}
    actions = corporate_actions or []
    rows, open_lots = _trade_rows_and_open_lots(
        _fill_records(fills), contract_multipliers=multipliers, corporate_actions=actions
    )
    dividends = sum(
        float(event.cash_amount) for event in actions if event.action_type is CorporateActionType.CASH_DIVIDEND
    )
    realized_gross_pnl = sum(float(row["gross_pnl"]) for row in rows)
    realized_costs = sum(float(row["costs"]) for row in rows)
    realized_pnl = realized_gross_pnl - realized_costs
    open_unrealized_pnl = _open_unrealized_pnl(open_lots, positions or [], multipliers)
    total_commission = sum(float(fill.commission) for fill in fills)
    total_slippage = sum(float(fill.slippage) for fill in fills)
    # Slippage is embedded in executed fill prices, so it already sits inside
    # gross/unrealized PnL; only commission is an additional cash cost. The
    # slippage total is reported as attribution, not charged again.
    total_pnl = realized_gross_pnl + open_unrealized_pnl - total_commission + dividends
    expected_final_equity = float(initial_equity) + total_pnl
    actual_final_equity = expected_final_equity if final_equity is None else float(final_equity)
    return {
        "fills": len(fills),
        "round_trips": len(rows),
        "open_lots": sum(len(lots) for lots in open_lots.values()),
        "realized_gross_pnl": realized_gross_pnl,
        "realized_costs": realized_costs,
        "realized_pnl": realized_pnl,
        "open_unrealized_pnl": open_unrealized_pnl,
        "commission": total_commission,
        "slippage": total_slippage,
        "dividends": dividends,
        "total_costs": total_commission,
        "total_pnl": total_pnl,
        "initial_equity": float(initial_equity),
        "expected_final_equity": expected_final_equity,
        "final_equity": actual_final_equity,
        "reconciliation_error": actual_final_equity - expected_final_equity,
    }


def trade_rows(fills: list[dict]) -> list[dict]:
    """Match fill dictionaries into round-trip trade rows."""
    rows, _ = _trade_rows_and_open_lots(fills)
    return rows


def _trade_rows_and_open_lots(
    fills: list[dict],
    *,
    contract_multipliers: dict[str, float] | None = None,
    corporate_actions: list[CorporateActionEvent] | None = None,
) -> tuple[list[dict], dict[str, deque[_Lot]]]:
    rows: list[dict] = []
    lots_by_symbol: dict[str, deque[_Lot]] = {}
    multipliers = {symbol.upper(): float(value) for symbol, value in (contract_multipliers or {}).items()}
    # Actions apply at a session's open, before any of its fills.
    timeline = sorted(
        [(event.timestamp, 0, i, event) for i, event in enumerate(corporate_actions or [])]
        + [(fill["timestamp"], 1, i, fill) for i, fill in enumerate(fills)],
        key=lambda item: item[:3],
    )
    for _, _, _, item in timeline:
        if isinstance(item, CorporateActionEvent):
            _apply_action(item, lots_by_symbol, rows, multipliers)
            continue
        fill = item
        symbol = fill["symbol"]
        side = fill["side"]
        quantity = float(fill["quantity"])
        multiplier = multipliers.get(symbol.upper(), 1.0)
        lot = _Lot(
            timestamp=fill["timestamp"],
            side=side,
            quantity=quantity,
            price=float(fill["price"]),
            commission_per_unit=_per_unit(fill.get("commission", 0.0), quantity),
        )
        lots = lots_by_symbol.setdefault(symbol, deque())
        while lot.quantity > QUANTITY_TOLERANCE and lots and lots[0].side != lot.side:
            open_lot = lots[0]
            matched_qty = min(open_lot.quantity, lot.quantity)
            rows.append(_matched_trade(symbol, open_lot, lot, matched_qty, multiplier=multiplier))
            open_lot.quantity -= matched_qty
            lot.quantity -= matched_qty
            if open_lot.quantity <= QUANTITY_TOLERANCE:
                lots.popleft()
        if lot.quantity > QUANTITY_TOLERANCE:
            lots.append(lot)
    return rows, lots_by_symbol


def _apply_action(
    event: CorporateActionEvent,
    lots_by_symbol: dict[str, deque[_Lot]],
    rows: list[dict],
    multipliers: dict[str, float],
) -> None:
    """Carry the open lots of ``event.symbol`` through one corporate action."""
    lots = lots_by_symbol.get(event.symbol)
    before = float(event.quantity_before)
    if not lots or not before or event.action_type is CorporateActionType.CASH_DIVIDEND:
        return
    multiplier = multipliers.get(event.symbol.upper(), 1.0)
    after = float(event.quantity_after)
    kind = event.action_type
    if kind in (CorporateActionType.SPLIT, CorporateActionType.STOCK_DIVIDEND):
        _rescale(lots, after / before, 0.0)
        return
    # Cash paid per share held: what a delisting or a merger's cash leg is worth.
    cash_per_share = float(event.cash_amount) / (before * multiplier)
    if kind is CorporateActionType.DELISTING or (kind is CorporateActionType.MERGER and not after):
        close = _Lot(
            timestamp=event.timestamp,
            side="sell" if lots[0].side == "buy" else "buy",
            quantity=sum(lot.quantity for lot in lots),
            price=cash_per_share,
            commission_per_unit=0.0,
        )
        while lots:
            open_lot = lots.popleft()
            rows.append(_matched_trade(event.symbol, open_lot, close, open_lot.quantity, multiplier=multiplier))
        return
    if kind is CorporateActionType.MERGER:
        # The cash leg returns part of the cost; the shares carry the rest.
        _rescale(lots, after / before, cash_per_share)
    if event.new_symbol:
        lots_by_symbol.setdefault(event.new_symbol, deque()).extend(lots_by_symbol.pop(event.symbol))


def _rescale(lots: deque[_Lot], ratio: float, cash_per_share: float) -> None:
    for lot in lots:
        lot.quantity *= ratio
        lot.price = (lot.price - cash_per_share) / ratio
        lot.commission_per_unit /= ratio


def trade_stats(trades) -> dict[str, float | int]:
    """Compute round-trip trade statistics from a Polars trade table."""
    frame = _to_polars_frame(trades)
    if frame.is_empty() or "pnl" not in frame.columns:
        return _empty_trade_stats()
    pnl = frame.select(pl.col("pnl").cast(pl.Float64, strict=False).alias("pnl")).drop_nulls()
    if pnl.is_empty():
        return _empty_trade_stats()
    wins = pnl.filter(pl.col("pnl") > 0)
    losses = pnl.filter(pl.col("pnl") < 0)
    gross_profit = float(wins.select(pl.col("pnl").sum()).item())
    gross_loss = abs(float(losses.select(pl.col("pnl").sum()).item()))
    trade_count = pnl.height
    win_count = wins.height
    loss_count = losses.height
    return {
        "trade_count": int(trade_count),
        "win_count": int(win_count),
        "loss_count": int(loss_count),
        "win_rate": float(win_count / trade_count),
        "total_pnl": float(pnl.select(pl.col("pnl").sum()).item()),
        "average_pnl": float(pnl.select(pl.col("pnl").mean()).item()),
        "profit_factor": gross_profit / gross_loss if gross_loss else 0.0,
    }


def _matched_trade(symbol: str, open_lot: _Lot, close_lot: _Lot, quantity: float, *, multiplier: float = 1.0) -> dict:
    direction = "long" if open_lot.side == "buy" else "short"
    gross_pnl = (
        (close_lot.price - open_lot.price) * quantity
        if direction == "long"
        else (open_lot.price - close_lot.price) * quantity
    ) * multiplier
    # Fill prices are executed prices with slippage already embedded (see
    # ExecutionHandler.apply_fill), so cash costs are commission only; charging
    # Fill.slippage here again would double-count it.
    costs = quantity * (open_lot.commission_per_unit + close_lot.commission_per_unit)
    return {
        "symbol": symbol,
        "direction": direction,
        "entry_time": open_lot.timestamp,
        "exit_time": close_lot.timestamp,
        "quantity": quantity,
        "entry_price": open_lot.price,
        "exit_price": close_lot.price,
        "gross_pnl": gross_pnl,
        "costs": costs,
        "pnl": gross_pnl - costs,
        "return": (gross_pnl - costs) / abs(open_lot.price * quantity * multiplier),
        "holding_period": close_lot.timestamp - open_lot.timestamp,
    }


def _open_unrealized_pnl(
    open_lots: dict[str, deque[_Lot]],
    positions: list[Position],
    contract_multipliers: dict[str, float],
) -> float:
    marks = {position.symbol.upper(): float(position.last_price) for position in positions}
    total = 0.0
    for symbol, lots in open_lots.items():
        mark = marks.get(symbol.upper())
        if mark is None:
            continue
        multiplier = contract_multipliers.get(symbol.upper(), 1.0)
        for lot in lots:
            if lot.side == "buy":
                total += (mark - lot.price) * lot.quantity * multiplier
            else:
                total += (lot.price - mark) * lot.quantity * multiplier
    return total


def _per_unit(value: float, quantity: float) -> float:
    return float(value) / quantity if quantity else 0.0


def _empty_trade_stats() -> dict[str, float | int]:
    return {
        "trade_count": 0,
        "win_count": 0,
        "loss_count": 0,
        "win_rate": 0.0,
        "total_pnl": 0.0,
        "average_pnl": 0.0,
        "profit_factor": 0.0,
    }


def _to_polars_frame(frame) -> pl.DataFrame:
    if isinstance(frame, pl.DataFrame):
        return frame
    if isinstance(frame, pl.LazyFrame):
        return frame.collect()
    if type(frame).__module__.split(".")[0] == "pandas":
        return pl.from_pandas(frame)
    return pl.DataFrame(frame)


def _fill_records(fills: list[Fill]) -> list[dict]:
    return [
        {
            "timestamp": fill.timestamp,
            "symbol": fill.symbol,
            "side": fill.side.value,
            "quantity": fill.quantity,
            "price": fill.price,
            "commission": fill.commission,
            "slippage": fill.slippage,
        }
        for fill in fills
    ]


TRADE_COLUMNS = [
    "symbol",
    "direction",
    "entry_time",
    "exit_time",
    "quantity",
    "entry_price",
    "exit_price",
    "gross_pnl",
    "costs",
    "pnl",
    "return",
    "holding_period",
]

PNL_RECONCILIATION_COLUMNS = [
    "fills",
    "round_trips",
    "open_lots",
    "realized_gross_pnl",
    "realized_costs",
    "realized_pnl",
    "open_unrealized_pnl",
    "commission",
    "slippage",
    "dividends",
    "total_costs",
    "total_pnl",
    "initial_equity",
    "expected_final_equity",
    "final_equity",
    "reconciliation_error",
]
