"""Options-native volatility strategies, defined once and run on any underlying.

A strategy is a *structure* (legs placed by delta at one expiry), an optional
volatility *gate* read from the chain itself, and an *exit*: at expiry, at a
profit take, or at a days-to-expiry threshold (`exit_dte`). The last is also
what makes a chain like DoltHub's usable, which lists each contract for only
two or three weeks and drops it about eleven days before it expires. Nothing here names
an underlying: the same candidate is evaluated on SPY, QQQ or anything else the
store holds, and its identity does not change with where it was run.

Planning is Polars over the chain store; only the contracts a plan actually
trades become `OptionMarketFrame` objects (see `examples/spy_short_put_study.py`
for why). Every decision at a session uses that session's own close snapshot
and the past; the engine fills it on the next frame.

Volatility features per session (`vol_features`):

- `atm_iv`: mean IV of the call and put nearest 0.5 delta at the expiry
  nearest 30 days;
- `hv`: annualised close-to-close volatility of the underlying over the last
  `hv_window` sessions;
- `iv_hv`: `atm_iv / hv`, above 1 when options price more movement than the
  underlying has shown (the variance risk premium);
- `iv_rank`: where today's `atm_iv` sits in its trailing `rank_window`
  sessions, 0 lowest to 1 highest.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from math import floor, sqrt
from typing import Any, Literal

import numpy as np
import polars as pl

from .chains import (
    CHAIN_KEY,
    CHAIN_SCHEMA,
    ChainTimePolicy,
    OptionChainStore,
    attach_lifecycle,
    expiry_events_from_underlying,
    option_chain_to_frames,
    option_contracts_from_chain,
)
from .contracts import OptionContract, OptionMarketFrame

EASTERN_CLOSE = ChainTimePolicy(quote_time=time(16, 0), expiry_time=time(16, 0), zone="America/New_York")


@dataclass(frozen=True)
class Leg:
    right: Literal["CALL", "PUT"]
    side: Literal[-1, 1]
    role: Literal["short", "wing", "atm"]


#: Legs by role: `short` legs sit at the candidate's `delta`, `wing` legs at its
#: `wing` delta further out, `atm` legs at 0.5. A wing always protects the short
#: leg of its own right, so it must be further out of the money.
STRUCTURES: dict[str, tuple[Leg, ...]] = {
    "short_put": (Leg("PUT", -1, "short"),),
    "bull_put_spread": (Leg("PUT", -1, "short"), Leg("PUT", 1, "wing")),
    "short_strangle": (Leg("PUT", -1, "short"), Leg("CALL", -1, "short")),
    "iron_condor": (Leg("PUT", -1, "short"), Leg("PUT", 1, "wing"),
                    Leg("CALL", -1, "short"), Leg("CALL", 1, "wing")),
    "long_straddle": (Leg("PUT", 1, "atm"), Leg("CALL", 1, "atm")),
}

GATE_FEATURES = ("iv_hv", "iv_rank")


@dataclass(frozen=True)
class VolGate:
    """Enter only while `feature` is above (or below) `threshold`."""

    feature: str
    rule: Literal["above", "below"]
    threshold: float

    def __post_init__(self) -> None:
        if self.feature not in GATE_FEATURES:
            raise ValueError(f"unknown gate feature {self.feature}")

    def describe(self) -> dict[str, Any]:
        return {"feature": self.feature, "rule": self.rule, "threshold": self.threshold}

    def admits(self, row: dict[str, Any]) -> bool:
        value = row.get(self.feature)
        if value is None or not np.isfinite(value):
            return False
        return value > self.threshold if self.rule == "above" else value < self.threshold


@dataclass(frozen=True)
class OptionCandidate:
    """One strategy: structure, placement, holding rule and gate. No underlying."""

    structure: str
    dte: int
    delta: float = 0.30
    wing: float = 0.10
    profit_take: float | None = None
    gate: VolGate | None = None
    exit_dte: int | None = None

    def __post_init__(self) -> None:
        if self.structure not in STRUCTURES:
            raise ValueError(f"unknown structure {self.structure}")
        if not 0 < self.wing < self.delta < 0.5:
            raise ValueError("require 0 < wing < delta < 0.5")
        if self.profit_take is not None and not 0 < self.profit_take < 1:
            raise ValueError("profit_take must be a fraction in (0, 1)")
        if self.exit_dte is not None and not 0 < self.exit_dte < self.dte:
            raise ValueError("exit_dte must be positive and below dte")

    def params(self) -> dict[str, Any]:
        """What identifies the strategy; only the fields its structure uses."""
        roles = {leg.role for leg in STRUCTURES[self.structure]}
        params: dict[str, Any] = {"dte": self.dte, "profitTake": self.profit_take, "exitDte": self.exit_dte}
        if "short" in roles:
            params["delta"] = self.delta
        if "wing" in roles:
            params["wing"] = self.wing
        return params

    def target_delta(self, leg: Leg) -> float:
        size = {"short": self.delta, "wing": self.wing, "atm": 0.5}[leg.role]
        return size if leg.right == "CALL" else -size


@dataclass(frozen=True)
class Position:
    """One planned holding: legs entered at `entered`, closed at `exit_at` or expiry."""

    entered: datetime
    expiry: datetime
    exit_at: datetime | None
    legs: tuple[tuple[str, int], ...]
    spot: float
    premium: float


def vol_features(store: OptionChainStore, underlying: str, *, start: datetime, end: datetime,
                 hv_window: int = 21, rank_window: int = 252, iv_dte: int = 30,
                 splits: Sequence[tuple[datetime, float]] = ()) -> pl.DataFrame:
    """Per session: spot, adjusted, atm_iv, hv, iv_hv and iv_rank, each from that session and earlier.

    Windows count trading days, not rows: a chain published three days a week
    (DoltHub) has returns spanning two or three days, and treating each as one
    day would understate `hv` by about a quarter and overstate `iv_hv`.

    `spot` is the traded (unadjusted) price, which strikes and sizing use;
    `adjusted` divides every close before a split's ex-date by its factor, so
    returns and `hv` see no jump on the split. Adjusting earlier closes by a
    later split is not look-ahead for a return: the ratio of two closes is the
    same whichever side of the split both are expressed on.
    """
    warmup = start - timedelta(days=int(rank_window * 1.6) + 10)
    chain = store.scan(underlyings=[underlying], start=warmup, end=end)
    spot = (chain.filter(pl.col("underlying_price").is_not_null())
            .group_by("quote_date").agg(pl.col("underlying_price").first().alias("spot")))
    near = (chain.filter(pl.col("implied_volatility").is_not_null() & (pl.col("implied_volatility") > 0)
                         & pl.col("delta").is_not_null() & pl.col("dte").is_between(iv_dte - 15, iv_dte + 30))
            .with_columns((pl.col("dte") - iv_dte).abs().alias("_dte_distance")))
    expiry = (near.sort("quote_date", "_dte_distance", "expiry").group_by("quote_date", maintain_order=True)
              .first().select("quote_date", "expiry"))
    atm = (near.join(expiry, on=["quote_date", "expiry"])
           .with_columns((pl.col("delta").abs() - 0.5).abs().alias("_atm"))
           .sort("quote_date", "right", "_atm", "strike")
           .group_by("quote_date", "right", maintain_order=True).first()
           .group_by("quote_date").agg(pl.col("implied_volatility").mean().alias("atm_iv"),
                                       pl.len().alias("_sides"))
           .filter(pl.col("_sides") == 2).drop("_sides"))
    table = spot.join(atm, on="quote_date", how="left").sort("quote_date").collect()
    days = table["quote_date"].to_list()
    closes = table["spot"].to_numpy().astype(float)
    for ex_date, factor in splits:
        closes[[i for i, day in enumerate(days) if day < ex_date]] /= factor
    hv, rank = _trading_day_windows(days, closes, table["atm_iv"].to_numpy().astype(float),
                                    hv_window=hv_window, rank_window=rank_window)
    return (table.with_columns(pl.Series("adjusted", closes), pl.Series("hv", hv, nan_to_null=True),
                               pl.Series("iv_rank", rank, nan_to_null=True))
            .with_columns((pl.col("atm_iv") / pl.col("hv")).alias("iv_hv"))
            .filter(pl.col("quote_date") >= start))


def _trading_day_windows(days: list[datetime], closes: np.ndarray, iv: np.ndarray, *, hv_window: int,
                         rank_window: int) -> tuple[np.ndarray, np.ndarray]:
    """`hv` and `iv_rank` over the last `hv_window` / `rank_window` weekdays, however sparse the rows.

    Variance is summed per return and divided by the weekdays the returns
    span, so a two-day return counts as two days of variance. Holidays are not
    removed from the count; on a daily series that is a few percent of a year.
    """
    position = np.busday_count(days[0].date(), [day.date() for day in days]) if days else np.array([], int)
    squared = np.concatenate([[np.nan], np.diff(np.log(closes)) ** 2])
    span = np.concatenate([[0], np.diff(position)]).astype(float)
    hv = np.full(len(days), np.nan)
    rank = np.full(len(days), np.nan)
    for i in range(1, len(days)):
        inside = (position[i] - position[:i + 1] < hv_window) & (np.arange(i + 1) > 0)
        inside &= position[i] - (position - span)[:i + 1] <= hv_window
        covered = span[:i + 1][inside].sum()
        if covered >= 0.75 * hv_window and inside.sum() >= 5:
            hv[i] = sqrt(252 * squared[:i + 1][inside].sum() / covered)
        recent = position[i] - position[:i + 1] < rank_window
        window = iv[:i + 1][recent]
        window = window[np.isfinite(window)]
        if np.isfinite(iv[i]) and window.size >= 20 and position[i] - position[:i + 1][recent][0] >= rank_window // 2:
            rank[i] = float((window < iv[i]).sum()) / (window.size - 1)
    return hv, rank


def picks(store: OptionChainStore, candidate: OptionCandidate, underlying: str, *, start: datetime,
          end: datetime, dte_window: int = 10, spread_cap: float = 0.5) -> pl.DataFrame:
    """Per session, the candidate's legs at one expiry, or no row when any leg is missing."""
    eligible = (store.scan(underlyings=[underlying], start=start, end=end)
                .filter(pl.col("dte").is_between(candidate.dte - dte_window, candidate.dte + dte_window)
                        & pl.col("delta").is_not_null() & pl.col("underlying_price").is_not_null()
                        & (pl.col("ask") > 0) & ((pl.col("ask") - pl.col("bid")) <= spread_cap * pl.col("mid"))))
    expiry = (eligible.with_columns((pl.col("dte") - candidate.dte).abs().alias("_d"))
              .sort("quote_date", "_d", "expiry").group_by("quote_date", maintain_order=True)
              .first().select("quote_date", "expiry"))
    at_expiry = eligible.join(expiry, on=["quote_date", "expiry"])
    legs = STRUCTURES[candidate.structure]
    table = None
    for index, leg in enumerate(legs):
        chosen = (at_expiry.filter((pl.col("right") == leg.right)
                                   & ((pl.col("bid") > 0) if leg.side < 0 else pl.lit(True)))
                  .with_columns((pl.col("delta") - candidate.target_delta(leg)).abs().alias("_d"))
                  .sort("quote_date", "_d", "strike", "option_symbol")
                  .group_by("quote_date", maintain_order=True).first()
                  .select("quote_date", "expiry", "underlying_price",
                          pl.col("option_symbol").alias(f"symbol{index}"),
                          pl.col("strike").alias(f"strike{index}"), pl.col("mid").alias(f"mid{index}")))
        table = chosen if table is None else table.join(
            chosen.drop("expiry", "underlying_price"), on="quote_date")
    assert table is not None
    result = table.sort("quote_date").collect()
    for right in ("PUT", "CALL"):
        short = [i for i, leg in enumerate(legs) if leg.right == right and leg.role == "short"]
        wing = [i for i, leg in enumerate(legs) if leg.right == right and leg.role == "wing"]
        if short and wing:
            further = (pl.col(f"strike{wing[0]}") < pl.col(f"strike{short[0]}") if right == "PUT"
                       else pl.col(f"strike{wing[0]}") > pl.col(f"strike{short[0]}"))
            result = result.filter(further)
    return result


def leg_quotes(store: OptionChainStore, underlying: str, symbols: list[str], *, start: datetime,
               end: datetime) -> dict[str, dict[datetime, float]]:
    """Mid by session for each symbol, for exits and quote-gap checks."""
    rows = (store.scan(underlyings=[underlying], start=start, end=end)
            .filter(pl.col("option_symbol").is_in(symbols))
            .select("option_symbol", "quote_date", "mid").collect())
    quotes: dict[str, dict[datetime, float]] = {}
    for symbol, day, mid in rows.iter_rows():
        quotes.setdefault(symbol, {})[day] = mid
    return quotes


def plan(candidate: OptionCandidate, chosen: pl.DataFrame, features: pl.DataFrame,
         quotes: dict[str, dict[datetime, float]], sessions: list[datetime], *,
         max_quote_age: timedelta = timedelta(days=5)) -> tuple[list[Position], dict[str, int]]:
    """Non-overlapping positions: enter when flat and admitted, exit at expiry or the profit take.

    A position is only taken if every leg stays quoted (no gap longer than
    `max_quote_age`) until it is closed and its expiry has a session to settle
    in: the engine refuses to mark a held contract at a price nobody published.
    """
    legs = STRUCTURES[candidate.structure]
    gate = {row["quote_date"]: row for row in features.iter_rows(named=True)}
    index = {day: i for i, day in enumerate(sessions)}
    positions: list[Position] = []
    dropped = {"gated": 0, "no_expiry_session": 0, "quote_gaps": 0}
    free_after: datetime | None = None
    for row in chosen.iter_rows(named=True):
        day = row["quote_date"]
        if free_after is not None and day <= free_after:
            continue
        if candidate.gate is not None and not candidate.gate.admits(gate.get(day, {})):
            dropped["gated"] += 1
            continue
        if row["expiry"] not in index:
            dropped["no_expiry_session"] += 1
            continue
        symbols = [(row[f"symbol{i}"], leg.side) for i, leg in enumerate(legs)]
        premium = -sum(side * row[f"mid{i}"] for i, (_, side) in enumerate(symbols))
        held = sessions[index[day]:index[row["expiry"]] + 1]
        exit_at = None
        if candidate.exit_dte is not None:
            # Close once the position is this close to expiry: the decision is
            # taken at that session's close and fills on the next one.
            exit_at = next((later for later in held[1:-1]
                            if (row["expiry"] - later).days <= candidate.exit_dte), None)
        if candidate.profit_take is not None:
            for later in held[1:-1]:
                if exit_at is not None and later >= exit_at:
                    break
                found = [quotes.get(symbol, {}).get(later) for symbol, _ in symbols]
                marks = [mark for mark in found if mark is not None]
                if len(marks) != len(found):
                    continue
                value = sum(side * mark for (_, side), mark in zip(symbols, marks))
                # Credit: kept share of the premium. Debit: gain on what was paid.
                gained = (premium + value) / premium if premium > 0 else (value + premium) / -premium
                if gained >= candidate.profit_take:
                    exit_at = later
                    break
        close = exit_at if exit_at is not None else row["expiry"]
        last_held = held[held.index(close) + 1] if exit_at is not None else close
        if _gapped([quotes.get(symbol, {}) for symbol, _ in symbols], day, last_held, max_quote_age,
                   expiry=exit_at is None):
            dropped["quote_gaps"] += 1
            continue
        positions.append(Position(day, row["expiry"], exit_at, tuple(symbols), row["underlying_price"], premium))
        free_after = close
    return positions, dropped


def _gapped(series: list[dict[datetime, float]], first: datetime, last: datetime, max_age: timedelta,
            *, expiry: bool) -> bool:
    for marks in series:
        quoted = sorted(day for day in marks if first <= day <= last)
        if not quoted or quoted[0] != first:
            return True
        # Held to expiry, the contract may stop quoting before its last session
        # (the settlement frame carries the fixing); closed early, it must be
        # quoted when the exit fills.
        if (any(b - a > max_age for a, b in zip(quoted, quoted[1:]))
                or last - quoted[-1] > (max_age if expiry else timedelta(0))):
            return True
    return False


def strategy(positions: list[Position], underlying: str, *, exposure: float, multiplier: float = 100):
    """The runtime side of a plan: size at entry from equity, exit on schedule, sell assigned stock.

    `exposure` is underlying notional per short (or bought) leg as a share of
    equity, so the same candidate means the same thing on a $50 and a $500
    underlying. Entries wait for a flat book; an exit that cannot fill is
    re-requested every frame.
    """
    entries = {p.entered: p for p in positions}
    exits = sorted((p.exit_at, p) for p in positions if p.exit_at is not None)

    def decide(frame: OptionMarketFrame, account) -> dict[str, float]:
        now = frame.timestamp
        orders: dict[str, float] = {}
        if account.positions.get(underlying, 0):
            orders[underlying] = 0.0
        held = {s for s, q in account.positions.items() if q and s != underlying}
        for exit_at, position in exits:
            if exit_at <= now:
                orders.update({symbol: 0.0 for symbol, _ in position.legs if symbol in held})
        position = entries.get(now)
        if position is not None and not held:
            count = floor(account.equity * exposure / (position.spot * multiplier))
            if count > 0:
                orders.update({symbol: float(side * count) for symbol, side in position.legs})
        return orders

    return decide


def session_list(store: OptionChainStore, underlying: str, *, start: datetime, end: datetime) -> list[datetime]:
    return sorted(store.scan(underlyings=[underlying], start=start, end=end)
                  .select(pl.col("quote_date").unique()).collect().to_series().to_list())


def market(store: OptionChainStore, underlying: str, positions: list[Position], *, start: datetime,
           end: datetime) -> tuple[tuple[OptionContract, ...], tuple[OptionMarketFrame, ...]]:
    """Frames for every session in the window: the traded contracts plus one anchor per session.

    The anchor (the near-the-money put, never traded) puts every session and
    its underlying close on the timeline, so every candidate run on this
    underlying is marked on the same instants and the return matrix is
    synchronised.
    """
    symbols = sorted({symbol for p in positions for symbol, _ in p.legs})
    chain = store.scan(underlyings=[underlying], start=start, end=end)
    anchors = (chain.filter((pl.col("right") == "PUT") & pl.col("underlying_price").is_not_null())
               .with_columns((pl.col("strike") - pl.col("underlying_price")).abs().alias("_atm"))
               .sort("quote_date", "_atm", "dte", "option_symbol")
               .group_by("quote_date", maintain_order=True).first().drop("_atm"))
    traded = chain.filter(pl.col("option_symbol").is_in(symbols))
    rows = (pl.concat([traded, anchors]).unique(subset=["quote_date", "option_symbol"], keep="first")
            .sort(list(CHAIN_KEY)).collect())
    if rows.is_empty():
        rows = pl.DataFrame(schema=dict(CHAIN_SCHEMA))
    contracts = option_contracts_from_chain(rows)
    events = expiry_events_from_underlying(rows, contracts, time_policy=EASTERN_CLOSE)
    frames = attach_lifecycle(option_chain_to_frames(rows, time_policy=EASTERN_CLOSE), events)
    return contracts, frames


def window(first: date, last: date) -> tuple[datetime, datetime]:
    return datetime.combine(first, time(0), UTC), datetime.combine(last, time(23, 59), UTC)
