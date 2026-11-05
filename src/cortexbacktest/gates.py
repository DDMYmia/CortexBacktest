"""Timing gates: a factor or macro series that decides whether a position may stand.

One definition for the strategy search and the shadow run. A gate is open while
the latest value known at the bar satisfies its rule, and closed before enough
is known. A factor value is known at its `known_at`; a macro value the day after
the vendor first published that vintage (ALFRED `realtime_start`), never at a
later receipt, so history loaded today still has its original timing.
"""

from __future__ import annotations

from bisect import bisect_right
from datetime import UTC, date, datetime, time, timedelta
from statistics import median

from .adapters.observation_file import observation_factors
from .contracts import Bar

GATE_RULES = ("positive", "above_median", "below_median")


def gate_series(document: dict, source: str) -> list[tuple[datetime, float]]:
    """(available time, value) pairs from a factor or macro observation file."""
    if source == "factor":
        return observation_factors(document).select("available_at", "value").rows()
    # At each moment the gate reads the newest observation date known then; a later
    # revision of an older date (or a vintage edge repeating it) is not "latest".
    points = sorted((datetime.combine(date.fromisoformat(row["realtimeStart"]) + timedelta(days=1), time.min, UTC),
                     row["observationDate"], row["value"])
                    for row in document["records"] if row["value"] is not None)
    series, newest = [], ""
    for available, observed, value in points:
        if observed >= newest:
            newest = observed
            series.append((available, value))
    return series


class SeriesGate:
    """Open while the latest value known at the bar satisfies the rule; closed before enough is known."""

    def __init__(self, observations: list[tuple[datetime, float]], rule: str, lookback: int) -> None:
        if rule not in GATE_RULES:
            raise ValueError(f"unknown gate rule {rule}")
        ordered = sorted(observations)
        self.times = [t for t, _ in ordered]
        self.values = [v for _, v in ordered]
        self.rule, self.lookback = rule, lookback

    def __call__(self, bar: Bar) -> bool:
        known = bisect_right(self.times, bar.timestamp)
        if known == 0:
            return False
        latest = self.values[known - 1]
        if self.rule == "positive":
            return latest > 0
        if known < self.lookback:
            return False
        middle = median(self.values[known - self.lookback:known])
        return latest > middle if self.rule == "above_median" else latest < middle

    def first_open(self) -> datetime | None:
        need = 1 if self.rule == "positive" else self.lookback
        return self.times[need - 1] if len(self.times) >= need else None
