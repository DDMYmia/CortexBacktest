from datetime import UTC, datetime, timedelta
import sqlite3

import pytest

from cortexbacktest import Bar, Order, Side, Strategy
from cortexbacktest.event.optimization import optimize_strategy
from cortexbacktest.experiments import ExperimentStore
from cortexbacktest.pipelines.robustness import CostScenario, run_robustness_study


class BuyQuantity(Strategy):
    def __init__(self, quantity=1):
        self.quantity = quantity
        self.done = False

    def on_bar(self, bar):
        if self.done:
            return []
        self.done = True
        return [Order(bar.symbol, bar.timestamp, Side.BUY, self.quantity)]


def bars(count=12):
    return [Bar("X", datetime(2026, 1, 1, tzinfo=UTC) + timedelta(days=i),
                100+i, 101+i, 99+i, 100+i, volume=100) for i in range(count)]


def test_interruption_resume_reuses_complete_cases(tmp_path):
    path = tmp_path / "search.sqlite"

    def interrupt(row):
        raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        optimize_strategy(bars(), BuyQuantity, quantity=[1, 2, 3], checkpoint_path=path, early_stop=interrupt)
    resumed = optimize_strategy(bars(), BuyQuantity, quantity=[1, 2, 3], checkpoint_path=path)
    complete = optimize_strategy(bars(), BuyQuantity, quantity=[1, 2, 3])
    assert resumed.rows == complete.rows
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT attempts FROM cases").fetchall() == [(1,), (1,), (1,)]


@pytest.mark.parametrize("change", ["data", "code", "grid", "seed"])
def test_checkpoint_identity_rejects_changed_inputs(tmp_path, change):
    kwargs = dict(quantity=[1, 2], checkpoint_path=tmp_path / "search.sqlite", code_version="v1", random_state=1)
    optimize_strategy(bars(), BuyQuantity, **kwargs)
    data = bars()
    if change == "data":
        data = data[:-1]
    elif change == "code":
        kwargs["code_version"] = "v2"
    elif change == "grid":
        kwargs["quantity"] = [1, 3]
    else:
        kwargs["random_state"] = 2
    with pytest.raises(ValueError, match="identity mismatch"):
        optimize_strategy(data, BuyQuantity, **kwargs)


@pytest.mark.parametrize("backend", ["thread", "process"])
@pytest.mark.parametrize("cache", [True, False])
def test_parallel_results_and_checkpoint_replay_equal_serial(tmp_path, backend, cache):
    grid = dict(quantity=[1, 2, 3, 4, 5])
    expected = optimize_strategy(bars(), BuyQuantity, **grid)
    path = tmp_path / "search.sqlite"
    parallel = optimize_strategy(bars(), BuyQuantity, **grid, parallel_backend=backend,
                                 max_workers=2, worker_cache=cache, checkpoint_path=path)
    assert parallel.execution["backend"] == backend
    assert parallel.rows == expected.rows
    assert optimize_strategy(bars(), BuyQuantity, **grid, checkpoint_path=path).rows == expected.rows


def test_failed_case_retry_uses_new_strategy_and_engine(tmp_path):
    instances = []

    class FailOnce(BuyQuantity):
        def on_bar(self, bar):
            if len(instances) == 1:
                raise RuntimeError("temporary")
            return super().on_bar(bar)

    def factory(quantity):
        strategy = FailOnce(quantity)
        instances.append(strategy)
        return strategy

    result = optimize_strategy(bars(), factory, quantity=[1], retries=1)
    assert len(instances) == 2
    assert result.rows == optimize_strategy(bars(), BuyQuantity, quantity=[1]).rows
    store = ExperimentStore(tmp_path / "failed.sqlite", {"test": 1})
    with pytest.raises(RuntimeError):
        store.run({"x": 1}, lambda: (_ for _ in ()).throw(RuntimeError("failed")))
    assert store.get({"x": 1}) is None
    assert store.run({"x": 1}, lambda: {"score": 1}) == {"score": 1}


def test_oos_prices_cannot_change_first_fold_training_selection():
    data = bars()
    changed = data[:6] + [Bar("X", bar.timestamp, 200, 200, 200, 200, volume=100) for bar in data[6:]]
    kwargs = dict(parameter_grid={"quantity": [1, 2]}, train_sessions=6, test_sessions=3,
                  costs=[CostScenario("baseline")], label_horizon=1)
    original = run_robustness_study(data, BuyQuantity, **kwargs)
    perturbed = run_robustness_study(changed, BuyQuantity, **kwargs)
    first = original.selections.row(0, named=True)
    altered = perturbed.selections.row(0, named=True)
    assert first["parameters_json"] == altered["parameters_json"]
    assert first["train_score"] == altered["train_score"]
    assert first["train_end"] < first["test_start"]
    # Every OOS strategy is cold: first test bar only submits an order, so no
    # inherited training position can earn that day's return.
    assert original.out_of_sample.group_by("fold", maintain_order=True).first()["return"].to_list() == [0.0, 0.0]
    assert original.cost_sensitivity["fills"].to_list() == [1, 1]
