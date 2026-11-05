"""Quote-driven options cashbook. No theoretical prices are used for fills.

Strategies return absolute target quantities; changes execute next frame as IOC.
Multi-leg orders are sequential and may be partially filled (not combo orders).
"""
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from hashlib import sha256
import json
import orjson
from math import floor, isfinite
from types import MappingProxyType
from typing import Callable, Iterable, Mapping, Protocol, cast

import polars as pl

from ..event.financing import SettlementModel
from ..event.manifest import engine_code_digest
from .contracts import OptionContract, OptionMarketFrame, OptionQuote, QuotePolicy, utc
from .costs import OptionCostModel, ResearchOptionCosts


class FinancingModel(Protocol):
    """Annual credit / debit / stock-borrow rates, read at a moment."""

    def rates(self, at: datetime) -> tuple[float, float, float]: ...


class OptionMarginModel(Protocol):
    def requirement(self, positions: Mapping[str, float], contracts: Mapping[str, OptionContract],
                    marks: Mapping[str, float]) -> float: ...


@dataclass(frozen=True, slots=True)
class ResearchOptionMargin:
    """Unnetted research collateral, NOT a broker/Reg-T/SPAN certification.

    Long premium at market value, naked short premium + max(alpha*S-OTM,
    beta*base); base=S for calls, K for puts. No spread/covered-call relief.
    """
    alpha: float = 0.20
    beta: float = 0.10
    stock_fraction: float = 0.50

    def __post_init__(self):
        if not all(isfinite(v) and 0 <= v <= 1 for v in
                   (self.alpha, self.beta, self.stock_fraction)):
            raise ValueError("margin fractions must be in [0,1]")

    def requirement(self, positions, contracts, marks) -> float:
        total = 0.0
        for symbol, quantity in positions.items():
            if not quantity:
                continue
            if symbol not in contracts:
                total += abs(quantity) * marks[symbol] * self.stock_fraction
                continue
            contract = contracts[symbol]
            premium = marks[symbol]
            if quantity > 0:
                total += quantity * contract.multiplier * premium
            else:
                spot = marks[contract.underlying]
                otm = max(contract.right.sign * (contract.strike - spot), 0)
                base = spot if contract.right == "CALL" else contract.strike
                total += -quantity * contract.multiplier * (
                    premium + max(self.alpha * spot - otm, self.beta * base))
        return total


@dataclass(frozen=True, slots=True)
class OptionBacktestConfig:
    initial_cash: float = 100000
    currency: str = "USD"
    spread_fraction: float = 1.0
    option_fee: float = 0.65
    stock_fee_bps: float = 0.0
    participation: float = 1.0
    require_sizes: bool = True
    debit_rate: float = 0.0
    credit_rate: float = 0.0
    stock_borrow_rate: float = 0.0
    exercise_threshold: float = 0.01
    costs: OptionCostModel | None = None
    quote_policy: QuotePolicy = field(default_factory=QuotePolicy)
    settlement_model: SettlementModel = field(default_factory=SettlementModel)
    margin: OptionMarginModel = field(default_factory=ResearchOptionMargin)
    #: A dated rate curve. Set it and the three flat rates above must stay at
    #: zero, so a run can never be financed at two rates at once.
    financing: FinancingModel | None = None
    maintenance_fraction: float = 1.0
    data_id: str = "unspecified"
    strategy_id: str = "unspecified"

    def __post_init__(self):
        positive = (self.initial_cash, self.participation, self.maintenance_fraction)
        nonnegative = (self.option_fee, self.stock_fee_bps, self.debit_rate,
                       self.credit_rate, self.stock_borrow_rate, self.exercise_threshold)
        if not all(isfinite(x) and x > 0 for x in positive):
            raise ValueError("capital/participation/maintenance must be positive finite")
        if not all(isfinite(x) and x >= 0 for x in nonnegative):
            raise ValueError("fees/rates/threshold must be nonnegative finite")
        if not isfinite(self.spread_fraction) or not 0 <= self.spread_fraction <= 1:
            raise ValueError("spread_fraction must be in [0,1]")
        if self.participation > 1 or self.maintenance_fraction > 1:
            raise ValueError("participation and maintenance_fraction cannot exceed 1")
        if self.financing is not None and (self.debit_rate, self.credit_rate,
                                           self.stock_borrow_rate) != (0.0, 0.0, 0.0):
            # Same reason the costs plugin owns all of its numbers: two places
            # to set one rate is how a run ends up financed at a rate nobody
            # chose, and financing can be most of the P&L.
            raise ValueError("with a financing model, debit_rate/credit_rate/stock_borrow_rate "
                             "must stay at 0: configure them on the model")
        if self.costs is not None and (self.spread_fraction, self.option_fee,
                                       self.stock_fee_bps) != (1.0, 0.65, 0.0):
            # Two places to set the same number is how a run ends up costing
            # something nobody chose. The plugin owns all of it or none of it.
            raise ValueError("with a costs plugin, spread_fraction/option_fee/stock_fee_bps "
                             "must stay at their defaults: configure them on the plugin")

    @property
    def cost_model(self) -> OptionCostModel:
        """The plugin, or one built from the flat fields for a config that has none."""
        return self.costs or ResearchOptionCosts(
            spread_fraction=self.spread_fraction, option_commission=self.option_fee,
            stock_commission_bps=self.stock_fee_bps)


@dataclass(frozen=True, slots=True)
class OptionAccountView:
    timestamp: datetime
    cash: float
    equity: float
    margin: float
    unsettled_cash: float
    positions: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class OptionBacktestResult:
    snapshots: pl.DataFrame
    fills: pl.DataFrame
    orders: pl.DataFrame
    lifecycle: pl.DataFrame
    attribution: Mapping[str, float]
    manifest: Mapping[str, object]

    def summary(self) -> dict[str, float | int | None]:
        equity = self.snapshots["equity"].to_numpy()
        import numpy as np
        initial = float(cast(float, self.manifest["initial_cash"]))
        nav = np.r_[initial, equity]
        peak = np.maximum.accumulate(nav)
        drawdown = nav / peak - 1
        returns = np.divide(np.diff(nav), nav[:-1],
                            out=np.full(len(equity), np.nan), where=nav[:-1] > 0)
        finite = returns if np.all(np.isfinite(returns)) else np.array([])
        # Each observation is a frame, not automatically a daily return.
        tail = finite[finite <= np.quantile(finite, 0.05)] if len(finite) else finite
        return {"initial_cash": initial, "final_equity": float(equity[-1]),
                "net_pnl": float(equity[-1] - initial),
                "total_return": float(equity[-1] / initial - 1),
                "max_drawdown": float(drawdown.min()),
                "frame_expected_shortfall_95": float(-tail.mean()) if len(tail) else None,
                "fills": self.fills.height,
                "margin_breach_frames": int(self.snapshots["margin_breach"].sum()),
                "max_margin": float(self.snapshots["margin"].to_numpy().max())}


OptionStrategy = Callable[[OptionMarketFrame, OptionAccountView], Mapping[str, float]]


def run_option_backtest(contracts: Iterable[OptionContract], frames: Iterable[OptionMarketFrame],
                        strategy: OptionStrategy, *,
                        config: OptionBacktestConfig | None = None) -> OptionBacktestResult:
    """Fresh deterministic run. Missing expiry fixing/held marks fail explicitly.

    Input is consumed once; only the current frame is exposed to the strategy.
    Caller closures can still leak future data and must be audited separately.
    """
    return _OptionRun(tuple(contracts), config or OptionBacktestConfig()).run(frames, strategy)


# The field order is a property of the class, not of each instance.
_QUOTE_FIELDS = tuple(OptionQuote.__dataclass_fields__)


class _OptionRun:
    def __init__(self, contracts, config):
        self.config = config
        self.contracts = {c.symbol: c for c in contracts}
        if len(self.contracts) != len(contracts):
            raise ValueError("duplicate option symbols")
        self.underlyings = {c.underlying for c in contracts}
        if self.underlyings & self.contracts.keys():
            raise ValueError("option/underlying symbol collision")
        if any(c.currency != config.currency for c in contracts):
            raise ValueError("multi-currency accounts require an FX accounting model")
        self.positions: dict[str, float] = {}
        self.quotes: dict[str, OptionQuote] = {}
        self.cash = config.initial_cash
        self.receivables: list[tuple[datetime, float]] = []
        self.asset_cash: dict[str, float] = {}
        self.costs = {"option_spread": 0.0, "stock_spread": 0.0,
                      "option_fees": 0.0, "stock_fees": 0.0,
                      "financing": 0.0, "dividends": 0.0}
        self.fill_rows: list[dict] = []
        self.order_rows: list[dict] = []
        self.life_rows: list[dict] = []
        self.snapshot_rows: list[dict] = []
        self.seen_lifecycle: set[tuple] = set()
        self.expired: set[str] = set()
        self.consumed: dict[tuple, float] = {}
        self.digest = sha256()

    def marks(self):
        return {key: quote.mid for key, quote in self.quotes.items()}

    def rates(self, at):
        """Credit, debit and stock-borrow rates for the accrual starting at `at`."""
        if self.config.financing is None:
            return (self.config.credit_rate, self.config.debit_rate,
                    self.config.stock_borrow_rate)
        rates = self.config.financing.rates(at)
        if len(rates) != 3 or not all(isfinite(rate) and rate >= 0 for rate in rates):
            raise ValueError("financing model returned invalid rates")
        return rates

    def multiplier(self, symbol):
        return self.contracts[symbol].multiplier if symbol in self.contracts else 1.0

    def unsettled(self, now):
        # `now` is one value; normalizing it inside the generator repeated the
        # conversion once per receivable, and this is called on every financing
        # knot of every frame.
        limit = utc(now)
        return sum(amount for at, amount in self.receivables if utc(at) > limit)

    def equity(self, positions=None, cash=None):
        positions = self.positions if positions is None else positions
        return (self.cash if cash is None else cash) + sum(
            q * self.quotes[s].mid * self.multiplier(s) for s, q in positions.items() if q)

    def margin(self, positions=None):
        value = self.config.margin.requirement(
            self.positions if positions is None else positions, self.contracts, self.marks())
        if not isfinite(value) or value < 0:
            raise ValueError("margin plugin returned invalid collateral")
        return value

    def flow(self, symbol, amount, settlement_at, now):
        self.cash += amount
        self.asset_cash[symbol] = self.asset_cash.get(symbol, 0.0) + amount
        if amount > 0 and utc(settlement_at) > utc(now):
            self.receivables.append((settlement_at, amount))

    def trade(self, symbol, requested, frame, submitted_at, *, forced=False):
        now = frame.timestamp
        current = self.positions.get(symbol, 0.0)
        closing = current * requested < 0 and abs(requested) <= abs(current)
        row = {"timestamp": now, "submitted_at": submitted_at, "symbol": symbol,
               "requested": requested, "filled": 0.0, "reason": "", "forced": forced}
        def reject(reason):
            row["reason"] = reason
            self.order_rows.append(row)
        if symbol in self.expired or (symbol in self.contracts and
                                      utc(now) >= utc(self.contracts[symbol].last_trade_at or
                                                      self.contracts[symbol].expiry)):
            reject("expired")
            return
        quote = frame.quotes.get(symbol)
        if quote is None:
            reject("missing_quote")
            return
        policy = self.config.quote_policy
        if symbol in self.underlyings:
            policy = replace(policy, min_open_interest=0)
        reason = policy.rejection(quote, now, closing=closing or forced)
        if reason:
            reject(reason)
            return
        if not forced and utc(quote.event_time) <= utc(submitted_at):
            reject("quote_not_after_decision")
            return
        size = quote.ask_size if requested > 0 else quote.bid_size
        if size is None and self.config.require_sizes:
            reject("missing_size")
            return
        key = (symbol, utc(quote.event_time), requested > 0)
        capacity = max(0.0, size * self.config.participation - self.consumed.get(key, 0)) if (
            size is not None) else abs(requested)
        quantity = min(abs(requested), capacity)
        if symbol in self.contracts:
            quantity = floor(quantity)
        quantity *= 1 if requested > 0 else -1
        if not quantity:
            reject("no_liquidity")
            return
        mult = self.multiplier(symbol)
        price, fee = self.config.cost_model.execution(
            symbol, quantity, quote, multiplier=mult, is_option=symbol in self.contracts)
        if not isfinite(price) or price < 0 or not isfinite(fee) or fee < 0:
            raise ValueError("cost plugin returned an invalid price or commission")
        cash_change = -quantity * mult * price
        settlement_at = self.config.settlement_model.settlement_time(now)
        projected = dict(self.positions)
        projected[symbol] = current + quantity
        required = self.margin(projected)
        pending = self.unsettled(now) + (max(cash_change, 0) if utc(settlement_at) > utc(now) else 0)
        capacity_after = self.equity(projected, self.cash + cash_change - fee) - pending
        reducing = current * quantity < 0 and abs(quantity) <= abs(current)
        if not forced and not reducing and capacity_after + 1e-9 < required:
            reject("initial_margin")
            return
        self.positions[symbol] = projected[symbol]
        self.flow(symbol, cash_change, settlement_at, now)
        self.cash -= fee
        category = "option" if symbol in self.contracts else "stock"
        spread = abs(quantity) * mult * abs(price - quote.mid)
        self.costs[category + "_spread"] += spread
        self.costs[category + "_fees"] += fee
        self.consumed[key] = self.consumed.get(key, 0.0) + abs(quantity)
        self.fill_rows.append({"timestamp": now, "submitted_at": submitted_at,
                               "quote_at": quote.event_time, "symbol": symbol,
                               "quantity": quantity, "price": price, "mid": quote.mid,
                               "fee": fee, "spread_cost": spread, "forced": forced})
        row["filled"] = quantity
        row["reason"] = "filled" if quantity == requested else "partial_ioc"
        self.order_rows.append(row)

    def lifecycle(self, frame):
        for event in frame.lifecycle:
            if event.symbol not in self.contracts:
                raise ValueError("unknown lifecycle contract")
            key = (event.symbol, utc(event.event_time), event.quantity)
            if key in self.seen_lifecycle or event.symbol in self.expired:
                raise ValueError("duplicate lifecycle event")
            self.seen_lifecycle.add(key)
            contract = self.contracts[event.symbol]
            expiry = event.quantity is None
            if expiry and utc(event.event_time) < utc(contract.expiry):
                raise ValueError("expiry fixing precedes contractual expiry")
            if not expiry and (contract.exercise_style != "american" or
                               utc(event.event_time) >= utc(contract.expiry)):
                raise ValueError("early exercise requires American option before expiry")
            if not expiry and event.exercise is False:
                raise ValueError("early non-exercise cannot remove a position")
            current = self.positions.get(event.symbol, 0.0)
            quantity = current if expiry else event.quantity
            assert quantity is not None
            if quantity and (current * quantity <= 0 or abs(quantity) > abs(current)):
                raise ValueError("exercise/assignment exceeds signed position")
            exercise = event.exercise if event.exercise is not None else (
                not expiry or contract.intrinsic(event.underlying_price) > 0 and
                (contract.settlement == "cash" or contract.intrinsic(event.underlying_price)
                 >= self.config.exercise_threshold))
            at = event.settlement_at or self.config.settlement_model.settlement_time(frame.timestamp)
            if exercise and quantity:
                if contract.settlement == "cash":
                    amount = quantity * contract.multiplier * contract.intrinsic(event.underlying_price)
                    self.flow(event.symbol, amount, at, frame.timestamp)
                else:
                    shares = quantity * contract.multiplier * contract.right.sign
                    self.positions[contract.underlying] = self.positions.get(contract.underlying, 0) + shares
                    amount = -shares * contract.strike
                    self.flow(contract.underlying, amount, at, frame.timestamp)
                    # Transfer intrinsic value between books at the supplied fixing.
                    transfer = shares * (event.underlying_price - contract.strike)
                    self.asset_cash[event.symbol] = self.asset_cash.get(event.symbol, 0) + transfer
                    self.asset_cash[contract.underlying] -= transfer
            # A zero is not a position. Every expiring contract in the universe
            # reaches this line, held or not, so keeping the zeros grew the book
            # to the size of the chain: a 2010-2023 SPY run ended with 170,877
            # entries, all zero, which made every per-frame walk of the book
            # O(contracts) and put a 3 GB `positions` column in the snapshots.
            remaining = current - quantity
            if remaining:
                self.positions[event.symbol] = remaining
            else:
                self.positions.pop(event.symbol, None)
            if expiry:
                self.expired.add(event.symbol)
            self.life_rows.append({"timestamp": frame.timestamp, "symbol": event.symbol,
                                   "quantity": quantity, "exercise": exercise,
                                   "kind": "expiry" if expiry else "early_exercise_assignment",
                                   "fixing": event.underlying_price})
        for symbol, quantity in self.positions.items():
            if quantity and symbol in self.contracts and utc(frame.timestamp) >= utc(
                    self.contracts[symbol].expiry):
                raise ValueError(f"missing expiry settlement for held contract {symbol}")

    def run(self, frames, strategy):
        pending: dict[str, float] = {}
        previous = None
        for frame in frames:
            now = frame.timestamp
            if previous is not None and utc(now) <= utc(previous):
                raise ValueError("frames must be strictly increasing in UTC")
            # orjson, not stdlib json: this canonicalizes every quote of every
            # frame, and at 21 quotes it measured 119us per frame against 35us
            # (3.4x) for the same content. The digest identifies the data a run
            # consumed, and it is only ever compared against digests from the
            # same engine build -- the manifest carries `code_digest` alongside
            # it -- so the serializer is an implementation choice, marked by the
            # `quote_options_v2` engine tag.
            self.digest.update(orjson.dumps({"timestamp": utc(now).isoformat(),
                "quotes": {s: {name: getattr(q, name) for name in _QUOTE_FIELDS}
                           for s, q in sorted(frame.quotes.items())},
                "lifecycle": [asdict(e) for e in frame.lifecycle],
                "dividends": dict(frame.dividends)}, default=str, option=orjson.OPT_SORT_KEYS))
            if previous is not None:
                # Split accrual at cash release instants so a weekend credit does
                # not earn interest before it settles.
                previous_utc, now_utc = utc(previous), utc(now)
                knots = sorted({previous_utc, now_utc} | {
                    at_utc for at, _ in self.receivables
                    if previous_utc < (at_utc := utc(at)) < now_utc})
                financing = 0.0
                short_value = sum(-q * self.quotes[s].mid for s, q in self.positions.items()
                                  if q < 0 and s in self.underlyings)
                for start, end in zip(knots, knots[1:]):
                    balance = self.cash - self.unsettled(start)
                    years = (end - start).total_seconds() / (365 * 86400)
                    credit, debit, borrow = self.rates(start)
                    rate = credit if balance >= 0 else debit
                    financing += (balance * rate - short_value * borrow) * years
                self.cash += financing
                self.costs["financing"] += financing
            self.receivables = [(at, amount) for at, amount in self.receivables if utc(at) > utc(now)]
            for symbol, quote in frame.quotes.items():
                if symbol not in self.contracts and symbol not in self.underlyings:
                    raise ValueError(f"unknown quote symbol {symbol}")
                old = self.quotes.get(symbol)
                if old is not None and utc(quote.event_time) < utc(old.event_time):
                    raise ValueError("quote observation time moved backwards")
                self.quotes[symbol] = quote
            for symbol, dividend in frame.dividends.items():
                if symbol not in self.underlyings:
                    raise ValueError("dividend on unknown underlying")
                amount = self.positions.get(symbol, 0) * dividend
                self.cash += amount
                self.costs["dividends"] += amount
            if previous is not None and any(utc(e.available_at) <= utc(previous)
                                            for e in frame.lifecycle):
                raise ValueError("lifecycle event delivered after its available frame")
            self.lifecycle(frame)
            needed = {s for s, q in self.positions.items() if q}
            needed |= {self.contracts[s].underlying for s in tuple(needed) if s in self.contracts}
            for symbol in needed:
                quote = self.quotes.get(symbol)
                if quote is None or self.config.quote_policy.rejection(quote, now, closing=True):
                    raise ValueError(f"missing/stale valuation quote for held asset {symbol}")
            breach = self.equity() - self.unsettled(now) < self.margin() * self.config.maintenance_fraction
            if breach:
                # Stable liquidation ordering, actual quotes/caps, no imaginary fills.
                for symbol in sorted(self.positions, key=lambda s: (s not in self.contracts, s)):
                    if self.positions[symbol]:
                        self.trade(symbol, -self.positions[symbol], frame, now, forced=True)
                    if self.equity() - self.unsettled(now) >= self.margin():
                        break
            if previous is not None:
                for symbol, target in sorted(pending.items()):
                    change = target - self.positions.get(symbol, 0)
                    if change:
                        if breach:
                            self.order_rows.append({"timestamp": now, "submitted_at": previous,
                                "symbol": symbol, "requested": change, "filled": 0.0,
                                "reason": "margin_call_cancelled", "forced": False})
                        else:
                            required_symbols = {symbol}
                            if symbol in self.contracts:
                                required_symbols.add(self.contracts[symbol].underlying)
                            if any(s not in self.quotes or self.config.quote_policy.rejection(
                                    self.quotes[s], now, closing=True) for s in required_symbols):
                                self.order_rows.append({"timestamp": now, "submitted_at": previous,
                                    "symbol": symbol, "requested": change, "filled": 0.0,
                                    "reason": "missing_valuation", "forced": False})
                            else:
                                self.trade(symbol, change, frame, previous)
            equity, margin = self.equity(), self.margin()
            view = OptionAccountView(now, self.cash, equity, margin, self.unsettled(now),
                                     MappingProxyType(dict(self.positions)))
            targets = dict(strategy(frame, view))
            for symbol, target in targets.items():
                if symbol not in self.contracts and symbol not in self.underlyings:
                    raise ValueError("target references unknown instrument")
                if not isfinite(target) or isinstance(target, bool):
                    raise ValueError("target quantity must be finite")
                if symbol in self.contracts and target != int(target):
                    raise ValueError("option targets must be integer contracts")
            pending = targets
            delta: float | None = sum(q for s, q in self.positions.items() if s in self.underlyings)
            # Total delta across distinct underlyings is not economically additive;
            # report it only for a single-underlying run.
            if len(self.underlyings) != 1:
                delta = None
            else:
                for s, q in self.positions.items():
                    if q and s in self.contracts:
                        d = self.quotes[s].delta
                        if d is None:
                            delta = None
                            break
                        assert delta is not None
                        delta += q * self.multiplier(s) * d
            self.snapshot_rows.append({"timestamp": now, "cash": self.cash, "equity": equity,
                "margin": margin, "unsettled_cash": view.unsettled_cash,
                "free_collateral": equity - margin - view.unsettled_cash,
                "margin_breach": breach or equity - view.unsettled_cash < margin * self.config.maintenance_fraction,
                "net_delta": delta,
                "positions": orjson.dumps(self.positions, option=orjson.OPT_SORT_KEYS).decode()})
            previous = now
        if previous is None:
            raise ValueError("at least one market frame required")
        for symbol, target in sorted(pending.items()):
            change = target - self.positions.get(symbol, 0)
            if change:
                self.order_rows.append({"timestamp": previous, "submitted_at": previous,
                    "symbol": symbol, "requested": change, "filled": 0.0,
                    "reason": "end_of_data", "forced": False})
        # Exact cashbook identity; gross market PnL is for the executed quantities,
        # NOT a counterfactual frictionless strategy or a Taylor Greek attribution.
        gross = {"option_market_pnl": 0.0, "stock_market_pnl": 0.0}
        for s in self.asset_cash.keys() | self.positions.keys():
            value = self.asset_cash.get(s, 0.0)
            if self.positions.get(s, 0):
                value += self.positions[s] * self.quotes[s].mid * self.multiplier(s)
            gross[("option" if s in self.contracts else "stock") + "_market_pnl"] += value
        for kind in ("option", "stock"):
            gross[kind + "_market_pnl"] += self.costs[kind + "_spread"]
        attribution = {**gross, **self.costs, "net_pnl": self.equity() - self.config.initial_cash}
        manifest = {"engine": "quote_options_v2", "code_digest": engine_code_digest(), "initial_cash": self.config.initial_cash,
            "data_digest": self.digest.hexdigest(), "data_id": self.config.data_id,
            "strategy_id": self.config.strategy_id, "contracts": [asdict(self.contracts[s]) for s in sorted(self.contracts)],
            "config": {k: (asdict(v) if hasattr(v, "__dataclass_fields__") else v)
                       for k, v in ((f, getattr(self.config, f)) for f in self.config.__dataclass_fields__)
                       if k not in ("margin", "financing")},
            "financing_model": type(self.config.financing).__qualname__ if (
                self.config.financing is not None) else "flat rates on the config",
            "financing_parameters": repr(self.config.financing) if (
                self.config.financing is not None) else "see config",
            "cost_model": type(self.config.cost_model).__qualname__,
            "cost_parameters": asdict(self.config.cost_model) if hasattr(
                self.config.cost_model, "__dataclass_fields__") else "custom: record externally",
            "margin_model": type(self.config.margin).__qualname__,
            "margin_parameters": asdict(self.config.margin) if hasattr(
                self.config.margin, "__dataclass_fields__") else "custom: record externally",
            "execution": "next-frame IOC; sequential legs; no synthetic liquidity"}
        manifest = json.loads(json.dumps(manifest, default=str, sort_keys=True))
        timestamp_type = pl.Datetime("us", "UTC")
        fills = pl.DataFrame(self.fill_rows, schema={
            "timestamp": timestamp_type, "submitted_at": timestamp_type, "quote_at": timestamp_type,
            "symbol": pl.String, "quantity": pl.Float64, "price": pl.Float64, "mid": pl.Float64,
            "fee": pl.Float64, "spread_cost": pl.Float64, "forced": pl.Boolean})
        orders = pl.DataFrame(self.order_rows, schema={
            "timestamp": timestamp_type, "submitted_at": timestamp_type, "symbol": pl.String,
            "requested": pl.Float64, "filled": pl.Float64, "reason": pl.String, "forced": pl.Boolean})
        lifecycle = pl.DataFrame(self.life_rows, schema={
            "timestamp": timestamp_type, "symbol": pl.String, "quantity": pl.Float64,
            "exercise": pl.Boolean, "kind": pl.String, "fixing": pl.Float64})
        return OptionBacktestResult(pl.DataFrame(self.snapshot_rows), fills,
            orders, lifecycle,
            MappingProxyType(attribution), MappingProxyType(manifest))
