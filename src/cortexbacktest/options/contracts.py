"""Point-in-time inputs. All timestamps are aware and compared in UTC."""
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from math import isfinite
from types import MappingProxyType
from typing import Literal, Mapping

from .pricing import OptionRight, option_right


def utc(value: datetime) -> datetime:
    """Reject naive timestamps; convert the rest to UTC.

    Called four times per quote per frame (twice validating the quote, twice
    placing it in a frame), so the already-at-UTC case skips `astimezone`: the
    instant is what every comparison, sort key and manifest digest here uses,
    and a zero-offset value already is that instant. Polars hands back
    `ZoneInfo("UTC")` rather than `timezone.utc`, and both render and compare
    identically, so the returned tzinfo object is not part of the contract.
    """
    offset = value.utcoffset() if value.tzinfo is not None else None
    if offset is None:
        raise ValueError("timezone-aware timestamps required")
    return value if not offset else value.astimezone(timezone.utc)


@dataclass(frozen=True, slots=True)
class OptionContract:
    symbol: str
    underlying: str
    right: OptionRight
    strike: float
    expiry: datetime
    multiplier: float = 100
    exercise_style: Literal["european", "american"] = "european"
    settlement: Literal["cash", "physical"] = "cash"
    currency: str = "USD"
    last_trade_at: datetime | None = None

    def __post_init__(self):
        expiry = utc(self.expiry)
        object.__setattr__(self, "expiry", expiry)
        if self.last_trade_at is not None:
            last_trade_at = utc(self.last_trade_at)
            if last_trade_at > expiry:
                raise ValueError("last trading time cannot follow expiry")
            object.__setattr__(self, "last_trade_at", last_trade_at)
        if not self.symbol or not self.underlying or self.symbol == self.underlying:
            raise ValueError("distinct nonempty option and underlying symbols required")
        object.__setattr__(self, "right", option_right(self.right))
        if self.exercise_style not in ("european", "american"):
            raise ValueError("invalid exercise style")
        if self.settlement not in ("cash", "physical"):
            raise ValueError("invalid settlement type")
        if not all(isfinite(x) and x > 0 for x in (self.strike, self.multiplier)):
            raise ValueError("positive finite strike/multiplier required")

    def intrinsic(self, spot: float) -> float:
        return max(self.right.sign * (spot - self.strike), 0)


@dataclass(frozen=True, slots=True)
class OptionQuote:
    """Also used for underlying bid/ask. Size is displayed units, not daily volume.

    All optional features must already be available at available_at; vendors must
    supply their actual publication time. No end-of-day volume inference is made.
    """
    symbol: str
    event_time: datetime
    available_at: datetime
    bid: float
    ask: float
    bid_size: float | None = None
    ask_size: float | None = None
    implied_volatility: float | None = None
    delta: float | None = None
    open_interest: float | None = None

    def __post_init__(self):
        # Normalize once here rather than at every read. Timestamps on a
        # constructed quote are UTC by class invariant, so the rest of the
        # option kernel can compare them directly: the conversion was measured
        # at four calls per quote per frame, a third of a chain backtest.
        event_time, available_at = utc(self.event_time), utc(self.available_at)
        if event_time > available_at:
            raise ValueError("quote cannot be available before observation")
        object.__setattr__(self, "event_time", event_time)
        object.__setattr__(self, "available_at", available_at)
        if not self.symbol or not (isfinite(self.bid) and isfinite(self.ask)):
            raise ValueError("finite quotes and symbol required")
        if self.bid < 0 or self.ask <= 0 or self.ask < self.bid:
            raise ValueError("invalid/crossed bid ask")
        for value in (self.bid_size, self.ask_size, self.implied_volatility, self.open_interest):
            if value is not None and (not isfinite(value) or value < 0):
                raise ValueError("sizes, IV and OI must be finite and nonnegative")
        if self.delta is not None and (not isfinite(self.delta) or abs(self.delta) > 1):
            raise ValueError("delta must be in [-1, 1]")

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2


@dataclass(frozen=True, slots=True)
class QuotePolicy:
    max_age: timedelta = timedelta(days=4)
    max_relative_spread: float = 1.0
    allow_zero_bid: bool = False
    min_open_interest: float = 0

    def __post_init__(self):
        if self.max_age < timedelta(0):
            raise ValueError("max_age must be nonnegative")
        if not all(isfinite(v) and v >= 0 for v in
                   (self.max_relative_spread, self.min_open_interest)):
            raise ValueError("finite nonnegative liquidity thresholds required")

    def rejection(self, quote: OptionQuote, now: datetime, *, closing: bool = False) -> str | None:
        # `now` is caller-supplied so it is still normalized here; the quote's
        # own instants are UTC by its class invariant and need no conversion.
        now_utc = utc(now)
        if quote.available_at > now_utc:
            return "unavailable"
        if now_utc - quote.event_time > self.max_age:
            return "stale_quote"
        # A risk-reducing close can sell at a genuine zero bid / wide spread.
        if not closing:
            if quote.bid == 0 and not self.allow_zero_bid:
                return "zero_bid"
            if (quote.ask - quote.bid) / quote.mid > self.max_relative_spread:
                return "wide_spread"
            if self.min_open_interest and (
                quote.open_interest is None or quote.open_interest < self.min_open_interest
            ):
                return "open_interest"
        return None


@dataclass(frozen=True, slots=True)
class OptionLifecycleEvent:
    """Explicit fixing/exercise/assignment; quantity None expires the whole position.

    An expiry must use the official settlement value, not a current stock proxy.
    Early quantity is signed like the position. exercise overrides the expiry rule.
    settlement_at is the actual cash/share settlement instant (economic exposure
    changes now, positive cash remains unavailable until settlement_at).
    """
    symbol: str
    event_time: datetime
    available_at: datetime
    underlying_price: float
    quantity: int | None = None
    exercise: bool | None = None
    settlement_at: datetime | None = None

    def __post_init__(self):
        event_time, available_at = utc(self.event_time), utc(self.available_at)
        if event_time > available_at:
            raise ValueError("lifecycle information available before observation")
        object.__setattr__(self, "event_time", event_time)
        object.__setattr__(self, "available_at", available_at)
        if not isfinite(self.underlying_price) or self.underlying_price < 0:
            raise ValueError("invalid settlement underlying price")
        if self.quantity is not None and (
            isinstance(self.quantity, bool) or not isinstance(self.quantity, int)
            or self.quantity == 0
        ):
            raise ValueError("exercise quantity must be a nonzero signed integer")
        if self.settlement_at is not None:
            settlement_at = utc(self.settlement_at)
            if settlement_at < available_at:
                raise ValueError("settlement cannot precede event availability")
            object.__setattr__(self, "settlement_at", settlement_at)


@dataclass(frozen=True, slots=True)
class OptionMarketFrame:
    timestamp: datetime
    quotes: Mapping[str, OptionQuote]
    lifecycle: tuple[OptionLifecycleEvent, ...] = ()
    dividends: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self):
        timestamp = utc(self.timestamp)
        object.__setattr__(self, "timestamp", timestamp)
        quotes = dict(self.quotes)
        # Quote and lifecycle timestamps are already UTC by their own invariant.
        for key, quote in quotes.items():
            if key != quote.symbol or quote.available_at > timestamp:
                raise ValueError("frame contains mismatched or future quote")
        if any(e.available_at > timestamp for e in self.lifecycle):
            raise ValueError("future lifecycle information")
        if any(not isfinite(x) or x < 0 for x in self.dividends.values()):
            raise ValueError("dividends must be finite positive amounts per share")
        object.__setattr__(self, "quotes", MappingProxyType(quotes))
        object.__setattr__(self, "dividends", MappingProxyType(dict(self.dividends)))
        object.__setattr__(self, "lifecycle", tuple(self.lifecycle))
