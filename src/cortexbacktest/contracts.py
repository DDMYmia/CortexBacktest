"""Core data contracts for the engine."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
from typing import NamedTuple
import math


# Below this, a share count or weight is treated as zero. Fills, position
# closes, rebalance deltas, and round-trip lot matching all need the same
# threshold: if the ledger closes a position that the trade report still
# considers open, PnL reconciliation breaks. Owning it in one place keeps those
# decisions from drifting apart.
QUANTITY_TOLERANCE = 1e-12


class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"


class TimeInForce(str, Enum):
    GTC = "gtc"
    DAY = "day"
    GTD = "gtd"


class OrderStatus(str, Enum):
    SUBMITTED = "submitted"
    ACCEPTED = "accepted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    REPLACED = "replaced"
    REJECTED = "rejected"
    CANCELLED = "cancelled"
    EXPIRED = "expired"

    @property
    def label(self) -> str:
        """Backtrader-style short lifecycle label for user-facing reports."""
        if self == OrderStatus.PARTIALLY_FILLED:
            return "partial"
        return self.value


class FillPrice(str, Enum):
    OPEN = "open"
    CLOSE = "close"
    MID = "mid"
    HLC3 = "hlc3"
    OHLC4 = "ohlc4"
    VWAP = "vwap"


@dataclass(frozen=True, slots=True)
class Tick:
    symbol: str
    timestamp: datetime
    last_price: float
    volume: float | None = None
    bid_price: float | None = None
    ask_price: float | None = None
    bid_volume: float | None = None
    ask_volume: float | None = None
    open_interest: float | None = None


@dataclass(frozen=True, slots=True)
class OrderBookLevel:
    price: float
    quantity: float


@dataclass(frozen=True, slots=True)
class OrderBookSnapshot:
    symbol: str
    timestamp: datetime
    bids: tuple[OrderBookLevel, ...]
    asks: tuple[OrderBookLevel, ...]

    @property
    def best_bid(self) -> OrderBookLevel | None:
        return self.bids[0] if self.bids else None

    @property
    def best_ask(self) -> OrderBookLevel | None:
        return self.asks[0] if self.asks else None

    @property
    def mid_price(self) -> float | None:
        if self.best_bid is None or self.best_ask is None:
            return None
        return (self.best_bid.price + self.best_ask.price) / 2.0


@dataclass(frozen=True, slots=True)
class OrderBookEvent:
    symbol: str
    timestamp: datetime
    side: Side
    price: float
    quantity: float
    action: str = "update"
    level: int = 0


@dataclass(frozen=True, slots=True)
class Contract:
    symbol: str
    exchange: str | None = None
    name: str | None = None
    product: str | None = None
    size: float = 1.0
    price_tick: float = 0.0
    min_volume: float = 1.0
    net_position: bool = True
    history_data: bool = True


class Bar(NamedTuple):
    """One OHLCV bar. Immutable; `copy.replace(bar, field=value)` makes a changed copy.

    A NamedTuple rather than a frozen dataclass: a frozen dataclass assigns
    each of the 13 fields through object.__setattr__, which on minute data
    was 43% of an idle engine run (0.69 us of about 1.1 us per bar); a tuple
    is built in C for about 0.1 us. See benchmarks/minute_partition_profile.py.
    """

    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float | None = None
    adj_close: float | None = None
    vwap: float | None = None
    open_interest: float | None = None
    halted: bool = False
    limit_up: float | None = None
    limit_down: float | None = None

    @property
    def return_close(self) -> float:
        return self.adj_close if self.adj_close is not None else self.close


class CorporateActionType(str, Enum):
    SPLIT = "split"
    CASH_DIVIDEND = "cash_dividend"
    STOCK_DIVIDEND = "stock_dividend"
    DELISTING = "delisting"
    SYMBOL_CHANGE = "symbol_change"
    MERGER = "merger"


@dataclass(frozen=True, slots=True)
class CorporateAction:
    """An event that changes the economics of a held position at ``ex_date`` open.

    - **Split** multiplies held quantity by ``ratio`` and divides the mark by the
      same factor, leaving equity unchanged: 2-for-1 is ``ratio=2.0``, a 1-for-5
      reverse split is ``ratio=0.2``.
    - **Cash dividend** pays ``amount`` per share into cash and leaves the
      position alone; a short position pays it out. Entitlement is fixed at
      ``ex_date`` on the shares held then; the cash arrives at ``pay_date`` if
      one is given, and at ``ex_date`` if not. ``withholding_tax`` is the
      fraction the issuer's jurisdiction keeps from a *received* dividend --
      a short pays the gross, because the borrower owes the lender the whole
      amount regardless of what any tax authority would have kept.
    - **Stock dividend** issues ``ratio`` extra shares per share held (a 10%
      stock dividend is ``ratio=0.1``). Economically it is a split by
      ``1 + ratio``, and it is modelled as one so equity is unchanged.
    - **Delisting** closes the position at ``amount`` per share — the final
      cash-out value, which is ``0.0`` for a worthless delisting. Leaving the
      position open instead would carry a stale mark forward for the rest of the
      backtest and inflate terminal equity, which is the single most common way
      a long backtest overstates its return.
    - **Symbol change** renames the position to ``new_symbol`` with no economic
      effect. Without it, later bars arrive under a symbol the account does not
      hold and the position is marked at its last pre-change price forever.
    - **Merger** exchanges the position for ``ratio`` shares of ``new_symbol``
      per share held, plus ``amount`` per share in cash consideration. A pure
      cash acquisition is a delisting with a non-zero ``amount``.

    These apply to *unadjusted* prices. Data carrying ``adj_close`` has already
    absorbed split and dividend effects, so combining the two double-counts; the
    engine rejects that combination rather than silently compounding it.
    """

    symbol: str
    ex_date: date
    action_type: CorporateActionType
    ratio: float = 1.0
    amount: float = 0.0
    new_symbol: str | None = None
    # Cash dividends only. None settles at ex-date, which is what every
    # backtest written before this field existed assumes.
    pay_date: date | None = None
    # Cash dividends only. None defers to the broker's default rate.
    withholding_tax: float | None = None

    def __post_init__(self) -> None:
        if not self.symbol:
            raise ValueError("corporate action requires a symbol")
        action_type = self.action_type
        if action_type != CorporateActionType.CASH_DIVIDEND:
            # Accepting a field that does nothing is how a backtest ends up
            # reporting a settlement date it never applied.
            if self.pay_date is not None:
                raise ValueError(f"{action_type.value} must not carry pay_date")
            if self.withholding_tax is not None:
                raise ValueError(f"{action_type.value} must not carry withholding_tax")

        if action_type == CorporateActionType.SPLIT:
            self._require_positive_ratio("split ratio")
            self._reject_amount("split")
            self._reject_new_symbol("split")
        elif action_type == CorporateActionType.CASH_DIVIDEND:
            if not math.isfinite(self.amount) or self.amount <= 0:
                raise ValueError("cash dividend amount must be positive and finite")
            self._reject_ratio("cash dividend")
            self._reject_new_symbol("cash dividend")
            if self.pay_date is not None and self.pay_date < self.ex_date:
                raise ValueError("cash dividend pay_date must not precede ex_date")
            if self.withholding_tax is not None and not 0.0 <= self.withholding_tax < 1.0:
                raise ValueError("withholding_tax must be in [0, 1)")
        elif action_type == CorporateActionType.STOCK_DIVIDEND:
            # Expressed as extra shares per share held, so 0 would be a no-op
            # and a negative value would destroy shares.
            self._require_positive_ratio("stock dividend ratio")
            self._reject_amount("stock dividend", "cash amount")
            self._reject_new_symbol("stock dividend")
        elif action_type == CorporateActionType.DELISTING:
            # Zero is meaningful here: a worthless delisting pays nothing.
            if not math.isfinite(self.amount) or self.amount < 0:
                raise ValueError("delisting amount must be zero or positive and finite")
            self._reject_ratio("delisting")
            self._reject_new_symbol("delisting")
        elif action_type == CorporateActionType.SYMBOL_CHANGE:
            self._require_new_symbol("symbol change")
            self._reject_ratio("symbol change")
            self._reject_amount("symbol change", "cash amount")
        else:  # MERGER
            self._require_new_symbol("merger")
            # ratio may be 0 when the consideration is entirely cash, but then
            # `amount` has to carry it or the position vanishes for nothing.
            if not math.isfinite(self.ratio) or self.ratio < 0:
                raise ValueError("merger ratio must be zero or positive and finite")
            if not math.isfinite(self.amount) or self.amount < 0:
                raise ValueError("merger cash consideration must be zero or positive and finite")
            if self.ratio == 0.0 and self.amount == 0.0:
                raise ValueError("merger must exchange shares, cash, or both")

        object.__setattr__(self, "symbol", self.symbol.upper())
        if self.new_symbol:
            object.__setattr__(self, "new_symbol", self.new_symbol.upper())

    def _require_positive_ratio(self, label: str) -> None:
        if not math.isfinite(self.ratio) or self.ratio <= 0:
            raise ValueError(f"{label} must be positive and finite")

    def _reject_ratio(self, label: str) -> None:
        if self.ratio != 1.0:
            raise ValueError(f"{label} must not carry a split ratio")

    def _reject_amount(self, label: str, noun: str = "dividend amount") -> None:
        if self.amount:
            raise ValueError(f"{label} must not carry a {noun}")

    def _require_new_symbol(self, label: str) -> None:
        if not self.new_symbol:
            raise ValueError(f"{label} requires new_symbol")

    def _reject_new_symbol(self, label: str) -> None:
        if self.new_symbol:
            raise ValueError(f"{label} must not carry new_symbol")


@dataclass(frozen=True, slots=True)
class CorporateActionEvent:
    """What a corporate action did to the account, for reporting."""

    timestamp: datetime
    symbol: str
    action_type: CorporateActionType
    quantity_before: float
    quantity_after: float
    # Net of withholding tax, and net of nothing else: this is what the account
    # is owed, not necessarily what its cash shows on this date.
    cash_amount: float = 0.0
    # Set when the position moved to another symbol (symbol change, merger), so
    # a reader can follow the holding across the rename instead of seeing it
    # disappear and a new one appear.
    new_symbol: str | None = None
    # Withholding kept from a received dividend. Positive when tax was taken;
    # zero everywhere else, including on a short, which owes the gross.
    tax_amount: float = 0.0
    # When the cash actually lands. Equal to the event's own date unless the
    # action carried a pay_date, in which case the amount is a receivable until
    # then and equity counts it while cash does not.
    settles_on: date | None = None


@dataclass(frozen=True, slots=True)
class ForcedLiquidation:
    """One position closed, in whole or in part, to answer a margin call."""

    symbol: str
    # Signed, as the position changed: negative closed a long, positive covered
    # a short. Reading it as "what the account traded" gets the sign backwards.
    quantity: float
    price: float
    commission: float = 0.0


@dataclass(frozen=True, slots=True)
class MarginCall:
    """A maintenance-margin breach and what was liquidated to answer it."""

    timestamp: datetime
    # Both measured at the moment of the breach, before any liquidation.
    equity: float
    requirement: float
    liquidations: tuple[ForcedLiquidation, ...] = ()
    # False when liquidating everything still left the account short of its
    # requirement -- the account is past the point where selling can fix it,
    # which is a distinct outcome from a call that was met and worth seeing.
    resolved: bool = True

    @property
    def shortfall(self) -> float:
        return self.requirement - self.equity


@dataclass(frozen=True, slots=True)
class FactorValue:
    symbol: str
    timestamp: datetime
    name: str
    value: float


@dataclass(frozen=True, slots=True)
class TargetWeight:
    symbol: str
    timestamp: datetime
    weight: float


@dataclass(frozen=True, slots=True)
class Order:
    symbol: str
    timestamp: datetime
    side: Side
    quantity: float
    order_type: OrderType = OrderType.MARKET
    limit_price: float | None = None
    stop_price: float | None = None
    trail_amount: float | None = None
    trail_percent: float | None = None
    tag: str | None = None
    id: str | None = None
    expires_at: datetime | None = None
    oco_group: str | None = None
    parent_id: str | None = None
    time_in_force: TimeInForce = TimeInForce.GTC


@dataclass(frozen=True, slots=True)
class OrderRecord:
    order: Order
    status: OrderStatus
    reason: str | None = None

    @property
    def status_label(self) -> str:
        """User-facing status label aligned with submitted/accepted/partial/filled."""
        return self.status.label


@dataclass(frozen=True, slots=True)
class Fill:
    symbol: str
    timestamp: datetime
    side: Side
    quantity: float
    price: float
    commission: float = 0.0
    slippage: float = 0.0
    order_id: str | None = None


@dataclass(slots=True)
class Position:
    """Account position.

    Deliberately mutable (not ``frozen``) so the ledger can update ``last_price``
    in place on every mark without reallocating one dataclass per bar, which is a
    measurable hot-path cost at millions of bars. It is never used as a dict key
    or set member; strategy-facing reads return copies.
    """

    symbol: str
    quantity: float
    last_price: float

    @property
    def market_value(self) -> float:
        return self.quantity * self.last_price
