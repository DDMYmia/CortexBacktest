"""What you should know about a bar dataset before trusting a result from it.

`validate_bars` answers "can this be used": nulls, duplicates, impossible OHLC,
a date outside the calendar. It runs on the hot path and rejects.

This answers a different question -- "what is in it" -- and rejects nothing. The
four findings here are the ones that changed conclusions on real data in this
repository, not a generic checklist:

- **Missing sessions**, and separately **short ones**. The minute store's SVB
  week is the case that matters and it is not a missing session: 2023-03-10 has
  bars, they just stop at 14:29. A session-count check sees nothing wrong, and
  aggregating those minutes to a daily bar erases the evidence entirely. Both
  are reported, because only one of them is visible after aggregation.
- **Stale prices.** A symbol whose close does not move for many sessions is
  usually halted, delisted-but-still-carried, or a vendor forward-fill. Returns
  computed across it are zero by construction, which flatters volatility and
  drawdown.
- **Outliers.** A 90% single-session move is either a real event or a bad print,
  and the difference is not visible from the number alone. Counting them tells a
  caller how much of the result rests on a handful of rows.
- **Symbol churn.** A universe that never loses a symbol is a survivorship
  warning, not a clean dataset. The option-chain layer learned this the same
  way: 24% of the 2020 universe is gone by 2023 in a store that carries
  delistings, and 0% would mean the store does not.

Findings carry counts and a few examples, following the shape
`option-chain-data-sources.md` uses: the measured fact, then what it costs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import polars as pl

from ..data.calendar import TradingCalendar
from .schema import STANDARD_BAR_COLUMNS

__all__ = ["BarDiagnostics", "diagnose_bars"]


@dataclass(frozen=True, slots=True)
class BarDiagnostics:
    """Descriptive findings about a bar dataset. Nothing here is an error."""

    sessions: int = 0
    symbols: int = 0
    missing_sessions: tuple[str, ...] = ()
    short_sessions: tuple[str, ...] = ()
    stale_runs: tuple[str, ...] = ()
    outliers: tuple[str, ...] = ()
    symbol_churn: float | None = None
    notes: tuple[str, ...] = ()

    @property
    def clean(self) -> bool:
        """No finding at all. Rare on real data, and worth being suspicious of."""
        return not (self.missing_sessions or self.short_sessions or self.stale_runs
                    or self.outliers or self.notes)

    def to_dict(self) -> dict:
        return {
            "sessions": self.sessions,
            "symbols": self.symbols,
            "missing_sessions": list(self.missing_sessions),
            "short_sessions": list(self.short_sessions),
            "stale_runs": list(self.stale_runs),
            "outliers": list(self.outliers),
            "symbol_churn": self.symbol_churn,
            "notes": list(self.notes),
        }

    def report(self) -> str:
        """One line per finding, for a log or the end of a study."""
        lines = [f"{self.sessions:,} sessions x {self.symbols:,} symbols"]
        for label, values in (
            ("missing sessions", self.missing_sessions),
            ("short sessions", self.short_sessions),
            ("stale price runs", self.stale_runs),
            ("outlier moves", self.outliers),
        ):
            if values:
                lines.append(f"{label}: {len(values)} -- {', '.join(values[:3])}"
                             + (" ..." if len(values) > 3 else ""))
        if self.symbol_churn is not None:
            lines.append(f"symbol churn: {self.symbol_churn:.1%} of first-session symbols absent by the last")
        lines.extend(self.notes)
        return "\n".join(lines)


def diagnose_bars(
    frame,
    *,
    calendar: TradingCalendar | None = None,
    stale_sessions: int = 5,
    outlier_return: float = 0.5,
    short_session_fraction: float = 0.5,
    max_examples: int = 8,
) -> BarDiagnostics:
    """Describe a normalized bar frame. Returns findings; raises only on shape.

    `stale_sessions` is how many consecutive unchanged closes count as stale,
    `outlier_return` the absolute session return that counts as extreme, and
    `short_session_fraction` the share of the median observation count below
    which a session is called truncated. All three are thresholds, not truths: a
    halted symbol and an illiquid one look identical here, a 50% move is a bad
    print about as often as it is news, and a half-length session is a real
    half day about as often as it is a feed that stopped.

    Short sessions are only visible while the frame is still intraday. One row
    per session means every session has the same count, so the check finds
    nothing -- correctly, because the evidence was aggregated away.
    """
    data = frame if isinstance(frame, pl.DataFrame) else pl.DataFrame(frame)
    missing = {"date", "symbol", "close"}.difference(data.columns)
    if missing:
        raise ValueError(f"diagnose_bars needs {sorted(missing)}; normalize the frame first")

    sessions = data["date"].dt.truncate("1d").unique().sort()
    symbols = data["symbol"].unique()
    notes: list[str] = []

    absent = _missing_sessions(sessions, calendar, max_examples)
    short = _short_sessions(data, short_session_fraction, max_examples)
    stale = _stale_runs(data, stale_sessions, max_examples)
    outliers = _outliers(data, outlier_return, max_examples)
    churn = _symbol_churn(data, sessions)

    if churn == 0.0 and len(sessions) > 60:
        notes.append(
            "no symbol leaves the universe across the sample: consistent with a "
            "survivorship-filtered dataset rather than a clean one"
        )
    if set(STANDARD_BAR_COLUMNS).difference(data.columns):
        notes.append("frame is not the standard bar schema; some checks were skipped")

    return BarDiagnostics(
        sessions=len(sessions),
        symbols=len(symbols),
        missing_sessions=absent,
        short_sessions=short,
        stale_runs=stale,
        outliers=outliers,
        symbol_churn=churn,
        notes=tuple(notes),
    )


def _missing_sessions(sessions: pl.Series, calendar: TradingCalendar | None,
                      max_examples: int) -> tuple[str, ...]:
    """Calendar sessions inside the sample's span that carry no bar at all."""
    if calendar is None or sessions.is_empty():
        return ()
    expected = calendar.between(sessions[0], sessions[-1]).dt.truncate("1d")
    present = set(sessions)
    gaps = [value for value in expected if value not in present]
    return tuple(value.date().isoformat() for value in gaps[:max_examples])


def _short_sessions(data: pl.DataFrame, fraction: float, max_examples: int) -> tuple[str, ...]:
    """Sessions carrying far fewer observations than the sample's median.

    Catches a feed that stopped mid-session, which a session-count check cannot
    see because the session is present. Silent on one-row-per-session data.
    """
    counts = (
        data.group_by(pl.col("date").dt.truncate("1d").alias("session"))
        .len()
        .sort("session")
    )
    if counts.height < 5 or counts["len"].max() == counts["len"].min():
        return ()
    median = float(cast(float | None, counts["len"].median()) or 0)
    if median <= 1:
        return ()
    cutoff = median * fraction
    thin = counts.filter(pl.col("len") < cutoff).sort("len")
    return tuple(
        f"{row['session'].date().isoformat()} {row['len']}/{median:.0f} observations"
        for row in thin.head(max_examples).iter_rows(named=True)
    )


def _stale_runs(data: pl.DataFrame, threshold: int, max_examples: int) -> tuple[str, ...]:
    """Symbols whose close repeats for `threshold` or more consecutive sessions."""
    if threshold < 2:
        return ()
    runs = (
        data.sort(["symbol", "date"])
        .with_columns(
            (pl.col("close") != pl.col("close").shift(1).over("symbol")).fill_null(True).alias("moved")
        )
        .with_columns(pl.col("moved").cum_sum().over("symbol").alias("run"))
        .group_by(["symbol", "run"])
        .agg(pl.len().alias("length"), pl.col("date").min().alias("from"))
        .filter(pl.col("length") >= threshold)
        .sort(["length", "symbol"], descending=[True, False])
    )
    return tuple(
        f"{row['symbol']} x{row['length']} from {row['from'].date().isoformat()}"
        for row in runs.head(max_examples).iter_rows(named=True)
    )


def _outliers(data: pl.DataFrame, threshold: float, max_examples: int) -> tuple[str, ...]:
    """Session-over-session moves beyond `threshold` in absolute terms."""
    moves = (
        data.sort(["symbol", "date"])
        .with_columns(
            (pl.col("close") / pl.col("close").shift(1).over("symbol") - 1.0).alias("move")
        )
        .drop_nulls("move")
        .filter(pl.col("move").abs() >= threshold)
        .sort("move", descending=True)
    )
    return tuple(
        f"{row['symbol']} {row['move']:+.1%} on {row['date'].date().isoformat()}"
        for row in moves.head(max_examples).iter_rows(named=True)
    )


def _symbol_churn(data: pl.DataFrame, sessions: pl.Series) -> float | None:
    """Fraction of first-session symbols absent from the last session."""
    if len(sessions) < 2:
        return None
    day = pl.col("date").dt.truncate("1d")
    first = set(data.filter(day == sessions[0])["symbol"])
    last = set(data.filter(day == sessions[-1])["symbol"])
    if not first:
        return None
    return len(first - last) / len(first)
