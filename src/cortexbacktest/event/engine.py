"""Transitional engine alias; use cortexbacktest.engine (removed in 0.3.0)."""
from ..runtime import BacktestEngine, BacktestResult, ResultRetention, SnapshotFrequency

__all__ = ["BacktestEngine", "BacktestResult", "ResultRetention", "SnapshotFrequency"]
