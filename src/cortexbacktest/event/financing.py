"""Optional financing and settlement models."""

from __future__ import annotations

from dataclasses import dataclass
from bisect import bisect_left, bisect_right
from datetime import date, datetime, timedelta
from hashlib import sha256
from math import isfinite
from typing import TYPE_CHECKING, Iterable
from zoneinfo import ZoneInfo

if TYPE_CHECKING:
    from ..data.calendar import TradingCalendar

from ..contracts import Position


@dataclass(frozen=True, slots=True)
class FinancingCharge:
    timestamp: datetime
    amount: float
    reason: str


@dataclass(frozen=True, slots=True)
class CashInterestModel:
    debit_rate: float = 0.0
    credit_rate: float = 0.0
    days_per_year: int = 365

    def __post_init__(self) -> None:
        _validate_financing(self.days_per_year, self.debit_rate, self.credit_rate)

    def charge(self, cash: float, days: int) -> float:
        if days <= 0:
            return 0.0
        rate = self.credit_rate if cash >= 0 else self.debit_rate
        return cash * rate * days / self.days_per_year


@dataclass(frozen=True, slots=True)
class ShortBorrowFeeModel:
    annual_rate: float = 0.0
    days_per_year: int = 365

    def __post_init__(self) -> None:
        _validate_financing(self.days_per_year, self.annual_rate)

    def charge(self, positions: dict[str, Position], days: int) -> float:
        if days <= 0:
            return 0.0
        short_value = sum(abs(position.market_value) for position in positions.values() if position.quantity < 0)
        return -short_value * self.annual_rate * days / self.days_per_year


@dataclass(frozen=True, slots=True)
class SettlementModel:
    """Natural-day delay by default; trading_days() uses explicit market sessions.

    Session dates are copied into the immutable model and included in manifests.
    The settlement wall time is preserved in the chosen market timezone.
    """
    delay_days: int = 0
    sessions: tuple[date, ...] | None = None
    timezone: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.delay_days, bool) or not isinstance(self.delay_days, int) or self.delay_days < 0:
            raise ValueError("delay_days must be non-negative")
        if self.sessions is not None:
            sessions = tuple(self.sessions)
            if not sessions or any(type(day) is not date for day in sessions) or tuple(sorted(set(sessions))) != sessions:
                raise ValueError("settlement sessions must be nonempty, sorted unique dates")
            object.__setattr__(self, "sessions", sessions)
        if self.timezone is not None:
            ZoneInfo(self.timezone)

    @classmethod
    def trading_days(cls, calendar: "TradingCalendar", *, delay_days: int = 1, timezone: str | None = None) -> "SettlementModel":
        return cls(delay_days, tuple(ts.date() for ts in calendar.sessions), timezone)

    def settlement_time(self, timestamp: datetime) -> datetime:
        if self.timezone is not None and timestamp.tzinfo is None:
            raise ValueError("timezone-aware timestamps required for market-timezone settlement")
        local = timestamp.astimezone(ZoneInfo(self.timezone)) if self.timezone else timestamp
        if self.sessions is None:
            settled = local + timedelta(days=self.delay_days)
        else:
            index = bisect_left(self.sessions, local.date())
            if index == len(self.sessions) or self.sessions[index] != local.date():
                raise ValueError("trade date is outside settlement calendar")
            target = index + self.delay_days
            if target >= len(self.sessions):
                raise ValueError("settlement calendar does not cover settlement date")
            day = self.sessions[target]
            settled = local.replace(year=day.year, month=day.month, day=day.day, fold=0)
        return settled.astimezone(timestamp.tzinfo) if timestamp.tzinfo else settled


@dataclass(slots=True)
class PendingCashSettlement:
    settles_at: datetime
    amount: float


@dataclass(slots=True)
class FinancingState:
    last_accrual_date: date | None = None
    charges: list[FinancingCharge] | None = None


@dataclass(frozen=True, slots=True, repr=False)
class RateSchedule:
    """Annual financing rates as a step function of the date, read at a moment.

    Rates are published daily and stand until the next publication, so a lookup
    takes the latest point at or before `at`. A moment before the first point
    raises: quietly financing part of a run at 0% is exactly the error this
    exists to prevent, and it is invisible in the result -- a flat 5% over
    2010-2023 put 105,219 of a 114,319 net P&L into `financing` and none of it
    was the strategy's.

    `credit` is earned on a positive balance, `debit` charged on a negative one,
    `borrow` charged on the value of short stock. All are decimals per annum
    (0.0525, not 5.25) and all are nonnegative.
    """

    dates: tuple[date, ...]
    credit: tuple[float, ...]
    debit: tuple[float, ...]
    borrow: tuple[float, ...]
    source: str = "unspecified"

    def __post_init__(self) -> None:
        dates = tuple(self.dates)
        if not dates:
            raise ValueError("a rate schedule needs at least one point")
        if any(type(day) is not date for day in dates) or tuple(sorted(set(dates))) != dates:
            raise ValueError("rate schedule dates must be sorted, unique and plain dates")
        object.__setattr__(self, "dates", dates)
        for name in ("credit", "debit", "borrow"):
            values = tuple(float(value) for value in getattr(self, name))
            if len(values) != len(dates):
                raise ValueError(f"{name} has {len(values)} values for {len(dates)} dates")
            if any(not isfinite(value) or value < 0 for value in values):
                raise ValueError(f"{name} rates must be finite and nonnegative")
            object.__setattr__(self, name, values)

    @classmethod
    def from_annual_percent(cls, points: Iterable[tuple[date, float]], *, source: str,
                            debit_spread: float = 0.0, borrow: float = 0.0) -> "RateSchedule":
        """Build from a published series quoted in percent, e.g. FRED `DGS1MO`.

        `debit_spread` is added to every point for the debit leg, because a broker
        charges more than it pays; `borrow` is a single flat stock-loan rate,
        since no free series publishes one per name.
        """
        rows = sorted(points)
        dates = tuple(day for day, _ in rows)
        credit = tuple(value / 100 for _, value in rows)
        return cls(dates, credit, tuple(rate + debit_spread for rate in credit),
                   (borrow,) * len(rows), source)

    def rates(self, at: datetime) -> tuple[float, float, float]:
        day = at.date()
        index = bisect_right(self.dates, day) - 1
        if index < 0:
            raise ValueError(f"{self.source} rate schedule starts {self.dates[0]}, "
                             f"after the {day} it was asked for")
        return self.credit[index], self.debit[index], self.borrow[index]

    @property
    def digest(self) -> str:
        payload = repr((self.source, self.dates, self.credit, self.debit, self.borrow))
        return sha256(payload.encode()).hexdigest()[:16]

    def __repr__(self) -> str:
        # Goes into run manifests, where the curve itself would be 3,500 rows of
        # noise; the digest identifies it and the source says where to get it.
        return (f"RateSchedule(source={self.source!r}, points={len(self.dates)}, "
                f"{self.dates[0]}..{self.dates[-1]}, digest={self.digest})")


def _validate_financing(days_per_year: int, *rates: float) -> None:
    if not isfinite(days_per_year) or days_per_year <= 0:
        raise ValueError("days_per_year must be finite and positive")
    if any(not isfinite(rate) or rate < 0 for rate in rates):
        raise ValueError("financing rates must be finite and non-negative")
