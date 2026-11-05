"""Pluggable, deterministic feedback-driven strategy search over finite grids."""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
import math
import random
from typing import Any, Protocol

import numpy as np

from .experiments import content_id

__all__ = ["SearchObservation", "SearchOptimizer", "SurrogateOptimizer"]


@dataclass(frozen=True, slots=True)
class SearchObservation:
    parameters: dict[str, Any]
    score: float


class SearchOptimizer(Protocol):
    """Stateless suggest API: reconstruct state from ordered observations.

    Plugins must be deterministic for a grid, seed, configuration and history.
    Return an unseen admissible case, or None to stop. configuration must be
    JSON serializable and include every setting that affects suggestions.
    """
    @property
    def configuration(self) -> Mapping[str, Any]: ...

    def suggest(
        self, grid: Mapping[str, Sequence[Any]], observations: Sequence[SearchObservation], *,
        random_state: int, constraint: Callable[[dict[str, Any]], bool] | None = None,
    ) -> dict[str, Any] | None: ...


@dataclass(frozen=True, slots=True)
class SurrogateOptimizer:
    """RBF-kernel surrogate with an uncertainty bonus and bounded candidate pool.

    Numeric dimensions are range-scaled; categorical dimensions use one-hot
    encoding. This is a budgeted heuristic, not a global-optimum guarantee.
    candidate_pool bounds acquisition memory; training uses the most recent
    max_observations plus the best case to bound the linear solve.
    """
    warmup: int = 5
    candidate_pool: int = 512
    exploration: float = 1.0
    length_scale: float = 0.3
    ridge: float = 1e-6
    max_observations: int = 128

    def __post_init__(self):
        for value in (self.warmup, self.candidate_pool, self.max_observations):
            if type(value) is not int or value < 1:
                raise ValueError("search sizes must be positive integers")
        if not math.isfinite(self.exploration) or self.exploration < 0:
            raise ValueError("exploration must be finite and non-negative")
        if any(not math.isfinite(value) or value <= 0 for value in (self.length_scale, self.ridge)):
            raise ValueError("length_scale and ridge must be finite and positive")

    @property
    def configuration(self) -> Mapping[str, Any]:
        return asdict(self)

    def suggest(self, grid, observations, *, random_state, constraint=None):
        names = list(grid)
        sizes = [len(grid[name]) for name in names]
        if any(size == 0 for size in sizes):
            raise ValueError("parameter grids must not contain empty iterables")
        total = math.prod(sizes)
        rng = random.Random(random_state + len(observations))
        indices = rng.sample(range(total), min(total, self.candidate_pool))
        candidates = []
        for index in indices:
            case = {}
            for name, size in reversed(list(zip(names, sizes, strict=True))):
                index, offset = divmod(index, size)
                case[name] = grid[name][offset]
            candidates.append(case)
        # Include grid neighbors of the incumbent even in very large spaces.
        if observations:
            best = max(observations, key=lambda item: item.score)
            for name in names:
                offset = list(grid[name]).index(best.parameters[name])
                for neighbor in (offset - 1, offset + 1):
                    if 0 <= neighbor < len(grid[name]):
                        candidates.append({**best.parameters, name: grid[name][neighbor]})
        seen = {content_id(item.parameters) for item in observations}
        available = []
        for case in candidates:
            key = content_id(case)
            if key not in seen and (constraint is None or constraint(dict(case))):
                available.append(case)
                seen.add(key)
        if not available:
            return None
        if len(observations) < self.warmup:
            return available[0]
        training = list(observations[-self.max_observations:])
        best = max(observations, key=lambda item: item.score)
        if best not in training:
            training = [best, *training[:-1]]

        def encode(cases):
            columns = []
            for name in names:
                values = grid[name]
                numeric = all(type(value) in (int, float) and math.isfinite(value) for value in values)
                if numeric:
                    lower, upper = min(values), max(values)
                    columns.append([(case[name] - lower) / (upper - lower or 1) for case in cases])
                else:
                    for value in values:
                        columns.append([float(case[name] == value) for case in cases])
            return np.asarray(columns, dtype=float).T if columns else np.zeros((len(cases), 1))

        x = encode([item.parameters for item in training])
        query = encode(available)

        def kernel(a, b):
            distance = ((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=2)
            return np.exp(-distance / (2 * self.length_scale ** 2))

        scores = np.asarray([item.score for item in training])
        scores = (scores - scores.mean()) / (scores.std() or 1.0)
        covariance = kernel(x, x) + self.ridge * np.eye(len(x))
        cross = kernel(query, x)
        mean = cross @ np.linalg.solve(covariance, scores)
        variance = np.maximum(0, 1 - np.sum(cross * np.linalg.solve(covariance, cross.T).T, axis=1))
        acquisition = mean + self.exploration * np.sqrt(variance)
        return available[int(np.argmax(acquisition))]
