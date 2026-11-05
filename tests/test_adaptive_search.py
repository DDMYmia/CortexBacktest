from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import sqlite3

import pytest

from cortexbacktest import Bar, Order, Side, Strategy
from cortexbacktest.event.optimization import optimize_strategy
from cortexbacktest.search import SearchObservation, SurrogateOptimizer


class PeakStrategy(Strategy):
    def __init__(self, x=0):
        self.x = x
        self.done = False

    def on_bar(self, bar):
        if self.done:
            return []
        self.done = True
        return [Order("X", bar.timestamp, Side.BUY, 1 + 100 - (self.x - 5.3)**2)]


def feed():
    return [Bar("X", datetime(2026, 1, 1, tzinfo=UTC) + timedelta(days=i), 100+i, 100+i, 100+i, 100+i) for i in range(3)]


def test_surrogate_finds_peak_with_a_fraction_of_exhaustive_runs():
    grid = [i / 10 for i in range(101)]
    full = optimize_strategy(feed(), PeakStrategy, x=grid)
    adaptive = optimize_strategy(feed(), PeakStrategy, x=grid, method="adaptive", max_tries=20, random_state=3)
    assert len(adaptive.rows) == 20
    assert adaptive.best_row()["final_equity"] >= full.best_row()["final_equity"] - 0.1
    assert adaptive.rows == optimize_strategy(feed(), PeakStrategy, x=grid, method="adaptive", max_tries=20, random_state=3).rows


def test_adaptive_resume_reconstructs_identical_observation_history(tmp_path):
    path = tmp_path / "adaptive.sqlite"
    kwargs = dict(x=[i / 10 for i in range(101)], method="adaptive", max_tries=15,
                  random_state=4, checkpoint_path=path)
    count = 0

    def interrupt(row):
        nonlocal count
        count += 1
        if count == 7:
            raise KeyboardInterrupt()
        return False

    with pytest.raises(KeyboardInterrupt):
        optimize_strategy(feed(), PeakStrategy, **kwargs, early_stop=interrupt)
    resumed = optimize_strategy(feed(), PeakStrategy, **kwargs)
    expected = optimize_strategy(feed(), PeakStrategy, **{**kwargs, "checkpoint_path": None})
    assert resumed.rows == expected.rows
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT max(attempts), count(*) FROM cases").fetchone() == (1, 15)
    with pytest.raises(ValueError, match="identity mismatch"):
        optimize_strategy(feed(), PeakStrategy, **kwargs, optimizer=SurrogateOptimizer(exploration=2))


@dataclass
class SequenceOptimizer:
    invalid: bool = False

    @property
    def configuration(self):
        return {"invalid": self.invalid}

    def suggest(self, grid, observations, *, random_state, constraint=None):
        if self.invalid:
            return {"x": 999}
        index = len(observations)
        return {"x": grid["x"][index]} if index < len(grid["x"]) else None


def test_custom_plugin_and_grid_guard():
    result = optimize_strategy(feed(), PeakStrategy, x=[1, 2], method="adaptive", optimizer=SequenceOptimizer())
    assert [row["x"] for row in result.rows] == [1, 2]
    with pytest.raises(ValueError, match="outside the grid"):
        optimize_strategy(feed(), PeakStrategy, x=[1, 2], method="adaptive", optimizer=SequenceOptimizer(True))


def test_constraints_and_categorical_parameters():
    optimizer = SurrogateOptimizer(warmup=1)
    observations = [SearchObservation({"mode": "a", "x": 1}, 1)]
    case = optimizer.suggest({"mode": ["a", "b"], "x": [1, 2]}, observations,
                             random_state=1, constraint=lambda case: case["mode"] == "b")
    assert case["mode"] == "b"
    assert optimizer.suggest({"x": [1, 2]}, [], random_state=1, constraint=lambda case: False) is None


def test_adaptive_explicit_parallelism_is_rejected():
    with pytest.raises(ValueError, match="sequential"):
        optimize_strategy(feed(), PeakStrategy, x=[1, 2], method="adaptive", max_workers=2, parallel_backend="process")


def test_plugin_cannot_repeat_cases_or_train_on_nonfinite_scores():
    class Repeating(SequenceOptimizer):
        def suggest(self, grid, observations, *, random_state, constraint=None):
            return {"x": 1}

    with pytest.raises(ValueError, match="repeated"):
        optimize_strategy(feed(), PeakStrategy, x=[1, 2], method="adaptive", optimizer=Repeating())
    with pytest.raises(ValueError, match="finite objective"):
        optimize_strategy(feed(), PeakStrategy, x=[1, 2], method="adaptive", maximize=lambda summary: float("nan"))
