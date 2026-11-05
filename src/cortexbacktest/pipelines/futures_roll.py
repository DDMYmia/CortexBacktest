"""Constant-contract futures exposure with real, close-first roll orders."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping

import polars as pl

from ..contracts import Bar, FillPrice, Order, OrderStatus, QUANTITY_TOLERANCE, Side
from ..data.futures import ContractRoll, _validate_roll_schedule
from ..engine import BacktestEngine, BacktestResult
from ..event.broker import BrokerSimulator
from ..event.feed import ColumnarBarFeed, market_slices_from_bars
from ..event.matching import MarketStateMatching
from ..event.risk import MarginRequirementRule
from ..event.strategy import Strategy
from ..reporting.summary import frame_records


@dataclass(frozen=True, slots=True)
class FuturesRollResult:
    event: BacktestResult
    rolls: pl.DataFrame

    def report(self) -> dict[str, Any]:
        return {"event": self.event.report(), "rolls": frame_records(self.rolls),
                "policy": "constant signed contract count; close old fully before opening new at a later bar",
                "assumptions": ["unadjusted individual-contract prices", "no expiry or physical delivery simulation",
                                "roll schedule supplied by caller; no future volume-based contract selection",
                                "different multipliers can change notional exposure", "unfinished rolls remain explicit"]}


class _RollStrategy(Strategy):
    def __init__(self, rolls: tuple[ContractRoll, ...], quantity: float):
        super().__init__()
        self.rolls = rolls
        self.quantity = quantity
        self.index = 0
        self.phase = "initial"
        self.pending: str | None = None
        self.records: list[dict[str, Any]] = [
            {"scheduled_at": roll.timestamp, "from_contract": roll.from_contract.upper(),
             "to_contract": roll.to_contract.upper(), "status": "scheduled",
             "quantity": 0.0, "close_filled_at": None, "open_filled_at": None}
            for roll in rolls]

    def on_before_bars(self, bars):
        if self.index >= len(self.rolls) or not bars:
            return []
        timestamp = bars[0].timestamp
        roll = self.rolls[self.index]
        symbols = {bar.symbol for bar in bars}
        old, new = roll.from_contract.upper(), roll.to_contract.upper()
        if self.phase == "initial" and self.pending is not None and timestamp >= roll.timestamp:
            self.broker.cancel_order(self.pending)
            self.pending = None
            self.phase = "holding"
        if self.pending is not None:
            return []
        if self.phase == "initial":
            if timestamp >= roll.timestamp:
                raise ValueError("feed needs an initial-contract bar before the first roll")
            if old not in symbols:
                return []
            return [self._order(old, timestamp, self.quantity, "initial")]
        record = self.records[self.index]
        if self.phase == "holding" and timestamp >= roll.timestamp:
            held = self.broker.state.positions.get(old)
            self.signed_quantity = held.quantity if held is not None else 0.0
            record["quantity"] = self.signed_quantity
            if abs(self.signed_quantity) < QUANTITY_TOLERANCE:
                record["status"] = "skipped_flat"
                self.index += 1
                return []
            record["status"] = "waiting_close"
            if old in symbols:
                self.phase = "closing"
                record["status"] = "closing"
                return [self._order(old, timestamp, -self.signed_quantity, "close")]
        elif self.phase == "opening":
            if new in symbols:
                record["status"] = "opening"
                return [self._order(new, timestamp, self.signed_quantity, "open")]
        return []

    def _order(self, symbol, timestamp, signed, leg):
        self.pending = f"roll-{self.index}-{leg}"
        return Order(symbol, timestamp, Side.BUY if signed > 0 else Side.SELL, abs(signed),
                     id=self.pending, tag=f"futures_roll:{leg}")

    def notify_order(self, record):
        if record.order.id != self.pending:
            return
        if record.status in {OrderStatus.REJECTED, OrderStatus.EXPIRED}:
            raise RuntimeError(f"futures roll order {record.order.id} {record.status.value}: {record.reason}")
        if record.status != OrderStatus.FILLED:
            return
        self.pending = None
        if self.phase == "initial":
            self.phase = "holding"
        elif self.phase == "closing":
            self.records[self.index].update(status="waiting_open", close_filled_at=self.timestamp)
            self.phase = "opening"
        elif self.phase == "opening":
            self.records[self.index].update(status="complete", open_filled_at=self.timestamp)
            self.index += 1
            self.phase = "holding"


def run_futures_roll_backtest(
    bars: Iterable[Bar] | ColumnarBarFeed, rolls: Iterable[ContractRoll], *,
    quantity: float = 1.0, initial_cash: float = 100_000,
    contract_multipliers: Mapping[str, float], commission_bps: float = 0.0,
    slippage_bps: float = 0.0, max_volume_share: float = 1.0,
    initial_margin: float = 0.1, build_manifest: bool = True,
) -> FuturesRollResult:
    """Run scheduled real-contract rolls with partial fills, costs and margin.

    Signed quantity allows long or short exposure. The old position must be
    completely closed before the new leg is submitted on a subsequent bar.
    This deliberately models the intervening exposure gap rather than assuming
    an atomic spread execution. Supply both contracts' raw bars around each roll.
    """
    schedule = tuple(sorted(rolls, key=lambda roll: roll.timestamp))
    if not schedule:
        raise ValueError("at least one roll is required")
    _validate_roll_schedule(list(schedule))
    if not math.isfinite(quantity) or quantity == 0:
        raise ValueError("quantity must be finite and nonzero")
    if not math.isfinite(initial_margin) or initial_margin <= 0:
        raise ValueError("initial_margin must be finite and positive")
    symbols = {roll.from_contract.upper() for roll in schedule} | {roll.to_contract.upper() for roll in schedule}
    multipliers = {symbol.upper(): value for symbol, value in contract_multipliers.items()}
    if not symbols.issubset(multipliers) or any(not math.isfinite(value) or value <= 0 for value in multipliers.values()):
        raise ValueError("positive contract multipliers required for every roll contract")
    feed = market_slices_from_bars(bars)
    for group in feed:
        for bar in group.bars:
            if bar.symbol not in symbols:
                raise ValueError(f"unexpected contract {bar.symbol}")
            if bar.adj_close is not None and bar.adj_close != bar.close:
                raise ValueError("real-contract execution requires unadjusted prices")
    strategy = _RollStrategy(schedule, quantity)
    broker = BrokerSimulator(initial_cash=initial_cash, fill_price=FillPrice.OPEN,
        futures_symbols=sorted(symbols), contract_multipliers=multipliers,
        commission_bps=commission_bps, slippage_bps=slippage_bps,
        max_volume_share=max_volume_share, matching_model=MarketStateMatching(),
        risk_rules=[MarginRequirementRule(initial_margin=initial_margin)])
    event = BacktestEngine(broker=broker, trade_on_close=False, build_manifest=build_manifest,
        run_metadata={"roll_policy": "close_first_constant_contracts", "quantity": quantity,
                      "rolls": [{"timestamp": roll.timestamp.isoformat(), "from": roll.from_contract,
                                 "to": roll.to_contract} for roll in schedule]}).run(feed, strategy)
    return FuturesRollResult(event, pl.DataFrame(strategy.records))
