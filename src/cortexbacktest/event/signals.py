"""Small signal helpers inspired by backtesting.py."""

from __future__ import annotations

from collections.abc import Sequence


def crossover(series1: Sequence[float], series2: Sequence[float] | float) -> bool:
    """Return True when series1 crosses above series2 on the latest point."""
    return _cross(series1, series2, direction=1)


def crossunder(series1: Sequence[float], series2: Sequence[float] | float) -> bool:
    """Return True when series1 crosses below series2 on the latest point."""
    return _cross(series1, series2, direction=-1)


def _cross(series1: Sequence[float], series2: Sequence[float] | float, *, direction: int) -> bool:
    if len(series1) < 2:
        return False
    left_prev = float(series1[-2])
    left_now = float(series1[-1])
    if isinstance(series2, (int, float)):
        right_prev = right_now = float(series2)
    else:
        if len(series2) < 2:
            return False
        right_prev = float(series2[-2])
        right_now = float(series2[-1])
    if direction > 0:
        return left_prev <= right_prev and left_now > right_now
    return left_prev >= right_prev and left_now < right_now
