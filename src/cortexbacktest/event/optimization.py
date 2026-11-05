"""Backtesting.py-style strategy parameter optimization."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from html import escape
from itertools import chain, islice, product
import random
import math
from pathlib import Path

from ..scheduling import SweepScheduling
from ..experiments import ExperimentStore, callable_identity, content_id
from ..search import SearchObservation, SearchOptimizer, SurrogateOptimizer
from typing import Any

import polars as pl

from .broker import BrokerSimulator
from .engine import BacktestEngine
from .feed import ColumnarBarFeed
from .parallel import ParallelBackend, ProcessPlan, run_cases, validate_backend, worker_feed
from .strategy import Strategy

StrategyFactory = Callable[..., Strategy]
ScoreFunc = Callable[[dict[str, float | int]], float]

#: Below this many cases, pool startup overhead is not worth paying — run
#: serial regardless of max_workers/parallel_backend.
DEFAULT_MIN_CASES_FOR_PARALLEL = 4

#: Cases in flight at once. Cases are generated lazily (see _parameter_cases),
#: so this — not the grid size — is what bounds memory: a search over a huge
#: parameter space never holds more than one batch of cases and results at a
#: time.
DEFAULT_BATCH_SIZE = 500

# Per-worker checkpoint state; event.parallel installs the feed itself.
_PROCESS_STORE: ExperimentStore | None = None
_PROCESS_RETRIES = 0


def _initialize_optimization_worker(checkpoint_path, identity, retries: int) -> None:
    global _PROCESS_STORE, _PROCESS_RETRIES
    _PROCESS_STORE = ExperimentStore(checkpoint_path, identity) if checkpoint_path else None
    _PROCESS_RETRIES = retries


@dataclass(frozen=True, slots=True)
class StrategyOptimizationResult:
    rows: list[dict[str, Any]]
    metric: str
    execution: dict[str, Any] = field(default_factory=dict)

    def heatmap_pl(self) -> pl.DataFrame:
        return pl.DataFrame(self.rows) if self.rows else pl.DataFrame()

    def heatmap_matrix_pl(self, *, x: str, y: str, value: str | None = None) -> pl.DataFrame:
        """Return a pivoted heatmap table for two optimized parameters."""
        metric = value or self.metric
        frame = self.heatmap_pl()
        if frame.is_empty():
            return frame
        for column in (x, y, metric):
            if column not in frame.columns:
                raise ValueError(f"unknown heatmap column: {column}")
        return frame.pivot(index=y, on=x, values=metric, aggregate_function="first").sort(y)

    def best_row(self) -> dict[str, Any] | None:
        if not self.rows:
            return None
        return max(self.rows, key=lambda row: row[self.metric])

    def best_params(self) -> dict[str, Any]:
        row = self.best_row()
        if row is None:
            return {}
        return {key: value for key, value in row.items() if key not in {"case", self.metric, "error"} and not key.startswith("summary_")}

    def report_html(self, *, max_rows: int = 100) -> str:
        rows = self.rows[:max_rows]
        columns = list(rows[0]) if rows else []
        header = "".join(f"<th>{escape(str(column))}</th>" for column in columns)
        body = "".join(
            "<tr>" + "".join(f"<td>{escape(str(row[column]))}</td>" for column in columns) + "</tr>"
            for row in rows
        )
        best = self.best_row() or {}
        return "\n".join(
            [
                "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">",
                "<title>CortexBacktest Strategy Optimization</title>",
                f"<style>{_OPTIMIZATION_CSS}</style></head><body><main>",
                "<h1>CortexBacktest Strategy Optimization</h1>",
                f"<section><h2>Best</h2><pre>{escape(str(best))}</pre></section>",
                f"<section><h2>Results</h2><table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table></section>",
                "</main></body></html>",
            ]
        )


def optimize_strategy(
    bars,
    strategy_factory: StrategyFactory,
    *,
    maximize: str | ScoreFunc = "final_equity",
    constraint: Callable[[dict[str, Any]], bool] | None = None,
    method: str = "grid",
    max_tries: int | None = None,
    random_state: int | None = None,
    max_workers: int = 1,
    parallel_backend: ParallelBackend = "auto",
    min_cases_for_parallel: int = DEFAULT_MIN_CASES_FOR_PARALLEL,
    batch_size: int = DEFAULT_BATCH_SIZE,
    early_stop: Callable[[dict[str, Any]], bool] | None = None,
    engine_factory: Callable[[], BacktestEngine] | None = None,
    broker_factory: Callable[[], BrokerSimulator] | None = None,
    scheduling: SweepScheduling | None = None,
    worker_cache: bool = True,
    checkpoint_path: str | Path | None = None,
    code_version: str | None = None,
    retries: int = 0,
    optimizer: SearchOptimizer | None = None,
    **params: Iterable[Any],
) -> StrategyOptimizationResult:
    """Search with retained pilot timing, bounded tasks and resumable case results.

    Auto runs one real case and estimates whether remaining work repays the
    configured process startup budget. The pilot result is retained. Custom
    engine/broker factories use serial in auto; explicit threads remain available.
    Explicit process mode rejects factories it cannot reproduce. Parallel results
    and early-stop decisions follow input order, independent of completion order.

    A checkpoint stores JSON summaries atomically and verifies data, strategy,
    parameter grid, objective, seed and code identity before reusing any result.
    Random strategy behavior must use an explicit seed in its constructor grid;
    random_state only seeds selection of search cases. code_version should identify
    external dependencies/global configuration not captured by source inspection.
    """
    if max_workers < 1 or min_cases_for_parallel < 1 or batch_size < 1 or retries < 0:
        raise ValueError("workers, minimum cases and batch size must be positive; retries non-negative")
    if method not in {"grid", "random", "adaptive"}:
        raise ValueError("method must be grid, random, or adaptive")
    if optimizer is not None and method != "adaptive":
        raise ValueError("optimizer requires method='adaptive'")
    if method == "adaptive" and parallel_backend in {"thread", "process"} and max_workers > 1:
        raise ValueError("adaptive search is sequential; use parallel_backend='auto' or 'serial'")
    if max_tries is not None and (type(max_tries) is not int or max_tries < 1):
        raise ValueError("max_tries must be a positive integer")
    validate_backend(parallel_backend)
    if parallel_backend == "process" and (engine_factory is not None or broker_factory is not None):
        raise ValueError("process optimization cannot use custom engine_factory/broker_factory; use serial or thread")
    policy = scheduling or SweepScheduling()
    metric = maximize if isinstance(maximize, str) else "score"
    grid = {name: list(values) for name, values in params.items()}
    count = math.prod(len(values) for values in grid.values())
    if max_tries is not None:
        count = min(count, max_tries)
    feed = bars if isinstance(bars, ColumnarBarFeed) else ColumnarBarFeed.from_bars(bars)
    cases = _parameter_cases(grid)
    if constraint is not None:
        cases = (case for case in cases if constraint(case))
    if method != "adaptive":
        cases = _select_cases(cases, method=method, max_tries=max_tries, random_state=random_state)
    search_optimizer = (optimizer or SurrogateOptimizer()) if method == "adaptive" else None
    identity = None
    store = None
    if checkpoint_path is not None:
        from .manifest import engine_code_digest
        identity = {
            "schema": 1, "data": feed.data_fingerprint().data_sha256, "grid": grid,
            "strategy": callable_identity(strategy_factory, code_version=code_version),
            "objective": maximize if isinstance(maximize, str) else callable_identity(maximize, code_version=code_version),
            "constraint": callable_identity(constraint, code_version=code_version),
            "engine_factory": callable_identity(engine_factory, code_version=code_version),
            "broker_factory": callable_identity(broker_factory, code_version=code_version),
            "engine_code": engine_code_digest(), "method": method,
            "max_tries": max_tries, "random_state": random_state, "code_version": code_version,
        }
        if search_optimizer is not None:
            identity["optimizer"] = {"class": callable_identity(type(search_optimizer), code_version=code_version),
                                     "configuration": dict(search_optimizer.configuration)}
        store = ExperimentStore(checkpoint_path, identity)

    def compute(case):
        engine = engine_factory() if engine_factory is not None else BacktestEngine(
            broker=broker_factory() if broker_factory is not None else None,
            snapshot_frequency="final", result_retention="summary", build_manifest=False)
        return _result_row(case, engine.run(feed, strategy_factory(**case), assume_sorted=True).summary(), maximize, metric)

    def run_case(case):
        return _checkpointed_case(case, lambda: compute(case), store, retries)

    rows = []
    execution: dict[str, Any] = {"requested_backend": parallel_backend, "workers": max_workers,
                               "startup_budget_seconds": policy.startup_seconds}
    if search_optimizer is not None:
        if any(not values for values in grid.values()):
            raise ValueError("parameter grids must not contain empty iterables")
        budget = min(count, max_tries if max_tries is not None else 50)
        observations: list[SearchObservation] = []
        seen = set()
        for _ in range(budget):
            # Copies prevent a plugin from mutating prior results or the grid.
            case = search_optimizer.suggest(deepcopy(grid), deepcopy(observations),
                                            random_state=random_state or 0, constraint=constraint)
            if case is None:
                break
            if set(case) != set(grid) or any(case[name] not in grid[name] for name in grid):
                raise ValueError("optimizer proposed parameters outside the grid")
            if constraint is not None and not constraint(dict(case)):
                raise ValueError("optimizer proposed a case rejected by the constraint")
            key = content_id(case)
            if key in seen:
                raise ValueError("optimizer proposed a repeated case")
            row = run_case(case)
            score = float(row[metric])
            if not math.isfinite(score):
                raise ValueError("adaptive search requires finite objective scores")
            seen.add(key)
            observations.append(SearchObservation(dict(case), score))
            rows.append(row)
            if early_stop is not None and early_stop(row):
                break
        return StrategyOptimizationResult(rows, metric, {**execution, "backend": "serial",
            "workers": 1, "method": "adaptive", "budget": budget, "evaluated": len(rows),
            "optimizer": type(search_optimizer).__name__})
    backend = parallel_backend
    peeked = list(islice(cases, min_cases_for_parallel))
    if max_workers == 1 or len(peeked) < min_cases_for_parallel:
        backend = "serial"
    # A filtered grid has only a conservative lower-bound count until consumed.
    remaining = max(0, (len(peeked) if constraint else count) - 1)
    custom = engine_factory is not None or broker_factory is not None
    process = None if custom else ProcessPlan(
        task=lambda case: (case, strategy_factory, maximize, metric), run=_run_case_process,
        setup=_initialize_optimization_worker, setup_args=(checkpoint_path, identity, retries), cache=worker_cache)
    results = run_cases(chain(peeked, cases), run_case, feed=feed, backend=backend, workers=max_workers,
                        remaining=remaining, policy=policy, execution=execution, process=process,
                        max_chunk=batch_size, max_inflight=batch_size)
    try:
        for row in results:
            rows.append(row)
            if early_stop is not None and early_stop(row):
                break
    finally:
        results.close()
    return StrategyOptimizationResult(rows, metric, execution)


def _checkpointed_case(case, compute, store, retries):
    if store is not None:
        return store.run(case, compute, retries=retries)
    for attempt in range(retries + 1):
        try:
            return compute()
        except Exception:
            if attempt == retries:
                raise
    raise AssertionError("unreachable")


def _result_row(case, summary, maximize, metric):
    return {"case": _case_name(case), **case, metric: _score(summary, maximize),
            **{f"summary_{key}": value for key, value in summary.items()}}


def _run_case_process(task):
    case, strategy_factory, maximize, metric = task
    feed = worker_feed()
    def compute():
        engine = BacktestEngine(snapshot_frequency="final", result_retention="summary", build_manifest=False)
        summary = engine.run(feed, strategy_factory(**case), assume_sorted=True).summary()
        return _result_row(case, summary, maximize, metric)
    return _checkpointed_case(case, compute, _PROCESS_STORE, _PROCESS_RETRIES)


def _parameter_cases(params: Mapping[str, Iterable[Any]]) -> Iterator[dict[str, Any]]:
    if not params:
        yield {}
        return
    keys = list(params)
    values = [list(params[key]) for key in keys]
    if any(not item for item in values):
        raise ValueError("parameter grids must not contain empty iterables")
    for combo in product(*values):
        yield dict(zip(keys, combo, strict=True))


def _select_cases(
    cases: Iterator[dict[str, Any]],
    *,
    method: str,
    max_tries: int | None,
    random_state: int | None,
) -> Iterator[dict[str, Any]]:
    if method == "grid":
        return cases if max_tries is None else islice(cases, max_tries)
    if max_tries is None:
        return cases
    return iter(_reservoir_sample(cases, max_tries, random_state))


def _reservoir_sample(items: Iterator[Any], k: int, random_state: int | None) -> list[Any]:
    """Algorithm R: a uniform sample of size k from a stream, without knowing its length up front."""
    rng = random.Random(random_state)
    reservoir: list[Any] = list(islice(items, k))
    for index, item in enumerate(items, start=k):
        replace_at = rng.randint(0, index)
        if replace_at < k:
            reservoir[replace_at] = item
    return reservoir


def _score(summary: dict[str, float | int], maximize: str | ScoreFunc) -> float:
    if callable(maximize):
        return float(maximize(summary))
    if maximize not in summary:
        raise ValueError(f"unknown optimization metric: {maximize}")
    return float(summary[maximize])


def _case_name(case: dict[str, Any]) -> str:
    if not case:
        return "default"
    return ",".join(f"{key}={case[key]}" for key in sorted(case))


_OPTIMIZATION_CSS = """
:root { color-scheme: light; }
body { margin: 0; background: #f7f8fa; color: #17202a; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
main { max-width: 1180px; margin: 0 auto; padding: 28px; }
table { width: 100%; border-collapse: collapse; background: white; border: 1px solid #dbe1ea; }
th, td { border-bottom: 1px solid #e7ebf1; padding: 8px 9px; text-align: left; font-size: 13px; }
th { background: #eef2f6; }
pre { background: white; border: 1px solid #dbe1ea; padding: 12px; overflow-x: auto; }
""".strip()
