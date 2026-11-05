"""Measured, bounded scheduling policy shared by sweep workflows."""
from __future__ import annotations

from collections import deque
from concurrent.futures import Executor
from dataclasses import dataclass
import math
from typing import Callable, Generator, Iterable, TypeVar

T = TypeVar("T")
R = TypeVar("R")


@dataclass(frozen=True, slots=True)
class SweepScheduling:
    """Conservative startup budget; auto uses one retained pilot case.

    startup_seconds is a configurable cost estimate, not a hardware guarantee.
    Explicit backends bypass auto selection. Tasks in flight are bounded.
    """
    startup_seconds: float = 1.0
    target_task_seconds: float = 0.1
    max_inflight_per_worker: int = 2

    def __post_init__(self):
        if not math.isfinite(self.startup_seconds) or self.startup_seconds < 0:
            raise ValueError("startup_seconds must be finite and non-negative")
        if not math.isfinite(self.target_task_seconds) or self.target_task_seconds <= 0:
            raise ValueError("target_task_seconds must be finite and positive")
        if self.max_inflight_per_worker < 1:
            raise ValueError("max_inflight_per_worker must be positive")

    def backend(self, case_seconds: float, remaining_cases: int, workers: int) -> str:
        saved = case_seconds * remaining_cases * (1 - 1 / max(workers, 1))
        return "process" if workers > 1 and saved > self.startup_seconds else "serial"

    def chunk_size(self, case_seconds: float, *, maximum: int) -> int:
        return max(1, min(maximum, int(self.target_task_seconds / max(case_seconds, 1e-6))))


def bounded_map(executor: Executor, function: Callable[[T], R], items: Iterable[T], *, limit: int) -> Generator[R, None, None]:
    """Stable input-order results with at most limit submitted tasks.

    Closing the iterator cancels work that has not started; running calls finish
    under the executor's normal shutdown semantics. No cases are replayed.
    """
    if limit < 1:
        raise ValueError("limit must be positive")
    iterator = iter(items)
    pending = deque()
    try:
        for _ in range(limit):
            try:
                pending.append(executor.submit(function, next(iterator)))
            except StopIteration:
                break
        while pending:
            yield pending.popleft().result()
            try:
                pending.append(executor.submit(function, next(iterator)))
            except StopIteration:
                pass
    finally:
        for future in pending:
            future.cancel()
