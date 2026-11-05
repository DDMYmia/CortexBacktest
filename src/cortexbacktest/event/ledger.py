"""Account ledger for event-driven backtests."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from math import isfinite
from typing import Callable, Iterable

from ..contracts import QUANTITY_TOLERANCE, Bar, Fill, Position, Side
from .financing import PendingCashSettlement, SettlementModel

EQUITY_SNAPSHOT_COLUMNS = (
    "timestamp",
    "equity",
    "cash",
    "pending_cash",
    "long_exposure",
    "short_exposure",
    "gross_exposure",
    "net_exposure",
    "leverage",
)


@dataclass(slots=True)
class AccountState:
    cash: float
    positions: dict[str, Position] = field(default_factory=dict)
    pending_cash: float = 0.0
    contract_multipliers: dict[str, float] = field(default_factory=dict)
    # Symbols settled by variation margin rather than by exchanging principal.
    # A futures position costs no cash to open, so its notional is not an asset
    # of the account; only the mark-to-market move since the last settlement is.
    futures_symbols: set[str] = field(default_factory=set)
    # Settlement basis per futures symbol: the price its P&L has been paid up
    # to. Equity counts only the move from here to the current mark.
    settlement_basis: dict[str, float] = field(default_factory=dict)

    def is_futures(self, symbol: str) -> bool:
        return symbol in self.futures_symbols

    def position_value(self, symbol: str, position: Position, price: float) -> float:
        """What a position contributes to equity at ``price``.

        Cash instruments contribute their full market value, because the account
        paid the principal to acquire them. Futures contribute only unsettled
        variation margin: the principal was never exchanged, so counting the
        notional would inflate equity by the whole contract size.
        """
        # Multipliers are registered under uppercase keys, so the lookup must
        # normalize too — otherwise equity() disagrees with equity_snapshot_row
        # for a position whose symbol was not already uppercase.
        multiplier = self.contract_multipliers.get(symbol.upper(), 1.0)
        if symbol in self.futures_symbols:
            basis = self.settlement_basis.get(symbol, position.last_price)
            return position.quantity * (price - basis) * multiplier
        return position.quantity * price * multiplier

    def maintenance_requirement(self, rate: float) -> float:
        """Margin the account must keep against its open positions at ``rate``.

        Measured on notional, not on what the positions contributed to equity.
        That distinction is the whole point for futures: a position that adds
        only its unsettled variation margin to equity still requires margin
        against the full contract value, which is how an account can be solvent
        on paper and still be liquidated.
        """
        multipliers = self.contract_multipliers
        total = 0.0
        for symbol, position in self.positions.items():
            multiplier = multipliers.get(symbol.upper(), 1.0) if multipliers else 1.0
            total += abs(position.quantity * position.last_price * multiplier)
        return total * rate

    def equity(self, prices: dict[str, float] | None = None) -> float:
        total = self.cash + self.pending_cash
        for symbol, position in self.positions.items():
            price = prices.get(symbol, position.last_price) if prices else position.last_price
            total += self.position_value(symbol, position, price)
        return total


class Ledger:
    """Owns cash, positions, fills, and account snapshots."""

    def __init__(
        self,
        *,
        initial_cash: float,
        retain_fills: bool = True,
        fill_sink: Callable[[Fill], None] | None = None,
        settlement_model: SettlementModel | None = None,
        contract_multipliers: dict[str, float] | None = None,
        futures_symbols: Iterable[str] | None = None,
    ) -> None:
        if not isfinite(initial_cash) or initial_cash <= 0:
            raise ValueError("initial_cash must be finite and positive")
        self.state = AccountState(cash=initial_cash)
        self.state.contract_multipliers.update(
            {symbol.upper(): float(multiplier) for symbol, multiplier in (contract_multipliers or {}).items()}
        )
        self.state.futures_symbols.update(symbol.upper() for symbol in (futures_symbols or ()))
        self.retain_fills = retain_fills
        self.fill_sink = fill_sink
        self.fills: list[Fill] = []
        self.fill_count = 0
        self.total_commission = 0.0
        self.total_slippage = 0.0
        self.settlement_model = settlement_model or SettlementModel()
        self.pending_settlements: list[PendingCashSettlement] = []

    def mark(self, bar: Bar) -> None:
        position = self.state.positions.get(bar.symbol)
        if position is not None:
            price = bar.adj_close if bar.adj_close is not None else bar.close
            if price != position.last_price:
                # Position is mutable by design; update in place to avoid
                # allocating one dataclass per bar (~10x faster on the mark path).
                position.last_price = price

    def contract_multiplier(self, symbol: str) -> float:
        return self.state.contract_multipliers.get(symbol.upper(), 1.0)

    def notional(self, symbol: str, quantity: float, price: float) -> float:
        return abs(quantity * price * self.contract_multiplier(symbol))

    def can_fill(self, *, side: Side, symbol: str, quantity: float, value: float, commission: float, allow_short: bool) -> str | None:
        if side == Side.BUY:
            # Futures exchange no principal, so notional is the wrong bar to
            # clear: one ES contract is ~$250k notional against ~$13k of margin.
            # Leverage for futures is enforced by MarginRequirementRule instead.
            required = commission if self.state.is_futures(symbol) else value + commission
            if self.state.cash < required:
                return "insufficient cash"
        if not allow_short and side == Side.SELL:
            current = self.state.positions.get(symbol)
            if current is None or current.quantity < quantity:
                return "short selling is disabled"
        return None

    def apply_fill(self, fill: Fill) -> None:
        # Resolve a fallible calendar lookup before mutating accounting totals.
        settles_at = None
        if fill.side == Side.SELL and self.settlement_model.delay_days > 0 and not self.state.is_futures(fill.symbol):
            settles_at = self.settlement_model.settlement_time(fill.timestamp)
        self.total_commission += fill.commission
        self.total_slippage += fill.slippage
        signed_qty = fill.quantity if fill.side == Side.BUY else -fill.quantity
        # Most runs register no contract multipliers; skip the per-fill
        # symbol.upper() allocation when the mapping is empty.
        multipliers = self.state.contract_multipliers
        multiplier = multipliers.get(fill.symbol.upper(), 1.0) if multipliers else 1.0

        if self.state.futures_symbols and self.state.is_futures(fill.symbol):
            self._apply_futures_fill(fill, signed_qty, multiplier)
        else:
            cash_delta = -signed_qty * fill.price * multiplier
            if self.settlement_model.delay_days > 0 and cash_delta > 0:
                assert settles_at is not None
                self.pending_settlements.append(
                    PendingCashSettlement(
                        settles_at=settles_at,
                        amount=cash_delta,
                    )
                )
                self.state.pending_cash += cash_delta
            else:
                self.state.cash += cash_delta
            self.state.cash -= fill.commission

            current = self.state.positions.get(fill.symbol)
            new_qty = signed_qty + (current.quantity if current else 0.0)
            if abs(new_qty) < QUANTITY_TOLERANCE:
                self.state.positions.pop(fill.symbol, None)
            else:
                self.state.positions[fill.symbol] = Position(fill.symbol, new_qty, fill.price)
        self.fill_count += 1
        if self.fill_sink is not None:
            self.fill_sink(fill)
        if self.retain_fills:
            self.fills.append(fill)

    def _apply_futures_fill(self, fill: Fill, signed_qty: float, multiplier: float) -> None:
        """Apply a futures fill: no principal changes hands, only realised P&L.

        Opening or adding costs nothing but commission. Reducing or closing pays
        out the P&L between the settlement basis and the fill price for the
        quantity that came off, because that part is now realised and will never
        be settled again by `settle_variation_margin`.
        """
        state = self.state
        symbol = fill.symbol
        current = state.positions.get(symbol)
        held = current.quantity if current else 0.0
        basis = state.settlement_basis.get(symbol, fill.price)

        closing = 0.0
        if held and (held > 0) != (signed_qty > 0):
            # Signs differ, so this fill reduces the position — possibly past
            # zero and out the other side.
            closing = min(abs(signed_qty), abs(held))
            direction = 1.0 if held > 0 else -1.0
            state.cash += direction * closing * (fill.price - basis) * multiplier

        state.cash -= fill.commission

        new_qty = held + signed_qty
        if abs(new_qty) < QUANTITY_TOLERANCE:
            state.positions.pop(symbol, None)
            state.settlement_basis.pop(symbol, None)
        else:
            if current is None or (held > 0) != (new_qty > 0):
                # New or flipped through zero: the position starts at this fill.
                state.settlement_basis[symbol] = fill.price
            elif abs(new_qty) > abs(held):
                # Added to the position. Blend the basis by quantity so the
                # unsettled P&L carried forward matches what was actually
                # unrealised on each lot.
                added = abs(signed_qty)
                state.settlement_basis[symbol] = (
                    basis * abs(held) + fill.price * added
                ) / (abs(held) + added)
            # A partial reduction keeps the basis: the remaining lots have not
            # been settled at any new price.
            state.positions[symbol] = Position(symbol, new_qty, fill.price)

    def settle_variation_margin(self) -> float:
        """Pay each futures position's mark-to-market move into cash.

        This is what makes a futures account a futures account: the position
        carries no principal, and the day's gain or loss is settled in cash
        against the settlement price. Returns the total settled, which is zero
        for an account holding no futures.

        Called at each session boundary, after positions are marked. Equity is
        unchanged by settlement — it moves value from unsettled P&L into cash,
        which is precisely why the basis has to advance with it.
        """
        state = self.state
        if not state.futures_symbols:
            return 0.0
        settled = 0.0
        for symbol, position in state.positions.items():
            if symbol not in state.futures_symbols:
                continue
            basis = state.settlement_basis.get(symbol, position.last_price)
            if basis == position.last_price:
                continue
            multiplier = state.contract_multipliers.get(symbol.upper(), 1.0)
            settled += position.quantity * (position.last_price - basis) * multiplier
            state.settlement_basis[symbol] = position.last_price
        state.cash += settled
        return settled

    def settle_cash(self, timestamp: datetime) -> float:
        settled = 0.0
        remaining = []
        for pending in self.pending_settlements:
            if pending.settles_at <= timestamp:
                settled += pending.amount
            else:
                remaining.append(pending)
        if settled:
            self.state.cash += settled
            self.state.pending_cash -= settled
        self.pending_settlements = remaining
        return settled

    def apply_cash_adjustment(self, amount: float) -> None:
        self.state.cash += amount

    def apply_split(self, symbol: str, ratio: float) -> tuple[float, float]:
        """Scale a held position by ``ratio``, keeping its market value constant.

        Returns ``(quantity_before, quantity_after)``. Fractional shares are kept
        rather than rounded: rounding here would leak value out of the account,
        and cash-in-lieu is a data question the caller models explicitly.
        """
        position = self.state.positions.get(symbol)
        if position is None:
            return 0.0, 0.0
        before = position.quantity
        position.quantity = before * ratio
        position.last_price = position.last_price / ratio
        return before, position.quantity

    def apply_cash_dividend(
        self,
        symbol: str,
        amount_per_share: float,
        *,
        withholding_tax: float = 0.0,
        pay_at: datetime | None = None,
    ) -> tuple[float, float, float]:
        """Credit ``amount_per_share`` per held share, net of withholding tax.

        Returns ``(quantity, net_amount, tax_amount)``. A short position is
        charged, which is what a borrower owes the lender -- and it is charged
        the *gross*: the tax is what a jurisdiction keeps from money paid out to
        a holder, and a short is not a holder. Applying it to both sides would
        turn a dividend into a profit centre for anyone short across the
        ex-date.

        With ``pay_at``, the amount becomes a receivable settling then, using
        the same pending-cash machinery as delayed trade settlement: equity
        counts it from the ex-date, cash does not move until the pay date. That
        gap is the point of the parameter -- a strategy that spends its dividend
        the moment it goes ex is spending money it does not have for a few
        weeks, and with T+0 dividends a backtest cannot see it doing so.
        """
        position = self.state.positions.get(symbol)
        if position is None:
            return 0.0, 0.0, 0.0
        multiplier = self.state.contract_multipliers.get(symbol.upper(), 1.0)
        gross = position.quantity * amount_per_share * multiplier
        tax = gross * withholding_tax if gross > 0 else 0.0
        net = gross - tax

        if pay_at is not None:
            self.pending_settlements.append(PendingCashSettlement(settles_at=pay_at, amount=net))
            self.state.pending_cash += net
        else:
            self.state.cash += net
        return position.quantity, net, tax

    def apply_delisting(self, symbol: str, amount_per_share: float) -> tuple[float, float]:
        """Close a position at ``amount_per_share`` and pay the proceeds in cash.

        Returns ``(quantity_before, cash_amount)``. ``0.0`` is a valid price: a
        worthless delisting pays nothing and the shares simply stop existing.
        Leaving the position open instead would mark it at its last trade for the
        rest of the run, which is how a backtest quietly keeps a bankrupt holding
        on the books.

        A short position pays cash out, because closing the borrow means buying
        the shares back at the cash-out value.
        """
        position = self.state.positions.pop(symbol, None)
        if position is None:
            return 0.0, 0.0
        multiplier = self.state.contract_multipliers.get(symbol.upper(), 1.0)
        cash_amount = position.quantity * amount_per_share * multiplier
        self.state.cash += cash_amount
        return position.quantity, cash_amount

    def apply_symbol_change(self, symbol: str, new_symbol: str) -> float:
        """Move a position to ``new_symbol`` with no economic effect.

        Returns the quantity carried over. If the account somehow already holds
        the destination symbol the two are merged by quantity, keeping the
        destination's mark: two tickers becoming one is what a symbol change
        means, and dropping either side would lose shares.
        """
        position = self.state.positions.pop(symbol, None)
        if position is None:
            return 0.0
        existing = self.state.positions.get(new_symbol)
        if existing is None:
            self.state.positions[new_symbol] = Position(new_symbol, position.quantity, position.last_price)
        else:
            existing.quantity += position.quantity
        return position.quantity

    def apply_merger(
        self,
        symbol: str,
        new_symbol: str,
        ratio: float,
        cash_per_share: float,
    ) -> tuple[float, float, float]:
        """Exchange a position for ``ratio`` shares of ``new_symbol`` plus cash.

        Returns ``(quantity_before, quantity_received, cash_amount)``. The
        received shares are marked at the acquired position's last price divided
        by ``ratio``, which keeps the share leg's market value continuous across
        the exchange — the same convention `apply_split` uses. The next bar for
        ``new_symbol`` re-marks it at the real price.
        """
        position = self.state.positions.pop(symbol, None)
        if position is None:
            return 0.0, 0.0, 0.0
        before = position.quantity
        multiplier = self.state.contract_multipliers.get(symbol.upper(), 1.0)
        cash_amount = before * cash_per_share * multiplier
        self.state.cash += cash_amount

        received = before * ratio
        if received:
            existing = self.state.positions.get(new_symbol)
            if existing is None:
                self.state.positions[new_symbol] = Position(
                    new_symbol, received, position.last_price / ratio
                )
            else:
                existing.quantity += received
        return before, received, cash_amount

    def equity_snapshot_row(self, timestamp: datetime) -> tuple:
        """Snapshot values ordered as EQUITY_SNAPSHOT_COLUMNS, without dict overhead."""
        timestamp, equity, cash, pending_cash, long_exposure, short_exposure = (
            self.equity_snapshot_core(timestamp)
        )
        gross_exposure = long_exposure - short_exposure
        return (
            timestamp, equity, cash, pending_cash, long_exposure, short_exposure,
            gross_exposure, long_exposure + short_exposure,
            gross_exposure / equity if equity else 0.0,
        )

    def equity_snapshot_core(self, timestamp: datetime) -> tuple:
        """The six measured snapshot values; the other three are derived from them.

        Gross, net and leverage are pure functions of long, short and equity, so
        a per-bar equity curve can store five columns and compute the rest once,
        vectorized, when the frame is built. That is three fewer float
        operations and three fewer array appends on every bar of every run.
        """
        state = self.state
        long_exposure = 0.0
        short_exposure = 0.0
        cash = state.cash
        pending_cash = state.pending_cash
        equity = cash + pending_cash
        positions = state.positions
        if positions:
            multipliers = state.contract_multipliers
            futures = state.futures_symbols
            basis_by_symbol = state.settlement_basis
            for position in positions.values():
                symbol = position.symbol
                market_value = position.quantity * position.last_price
                if multipliers:
                    market_value *= multipliers.get(symbol.upper(), 1.0)
                if futures and symbol in futures:
                    # A futures position holds no principal, so only the
                    # unsettled move counts toward equity. Exposure still
                    # reports the full notional: that is what the position is
                    # economically exposed to, and what leverage is measured on.
                    basis = basis_by_symbol.get(symbol, position.last_price)
                    multiplier = multipliers.get(symbol.upper(), 1.0) if multipliers else 1.0
                    equity += position.quantity * (position.last_price - basis) * multiplier
                else:
                    equity += market_value
                if market_value > 0:
                    long_exposure += market_value
                else:
                    short_exposure += market_value
        return (timestamp, equity, cash, pending_cash, long_exposure, short_exposure)

    def equity_snapshot(self, timestamp: datetime) -> dict[str, float | datetime]:
        return dict(zip(EQUITY_SNAPSHOT_COLUMNS, self.equity_snapshot_row(timestamp)))
