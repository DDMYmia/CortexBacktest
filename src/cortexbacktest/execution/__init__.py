"""Research-to-execution bridge subsystem."""

from .sweeps import ExecutionSweepResult, run_execution_sweep

__all__ = [
    "ExecutionSweepResult",
    "run_execution_sweep",
]
