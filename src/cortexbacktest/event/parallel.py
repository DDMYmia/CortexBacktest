"""Many backtests over one feed: serial, threads, or worker processes.

The one place that decides how a sweep runs. It picks the backend (auto keeps a
timed pilot case), ships the feed to process workers once per worker rather
than once per case, chunks tasks, bounds how many are in flight, and falls back
to serial where process startup is unavailable. Results follow input order.
"""

from __future__ import annotations

from collections.abc import Callable, Generator, Iterable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import batched
from multiprocessing import get_context
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Any, Literal

from ..scheduling import SweepScheduling, bounded_map
from .feed import ColumnarBarFeed
from .feed_storage import read_feed_ipc, write_feed_ipc
from .parquet_feed import ParquetBarFeed

ParallelBackend = Literal["serial", "thread", "process", "auto"]
BACKENDS = ("serial", "thread", "process", "auto")

_FEED: ColumnarBarFeed | None = None


@dataclass(frozen=True, slots=True)
class ProcessPlan:
    """How cases cross into worker processes.

    ``task`` turns a case into a picklable task in the parent; ``run`` is a
    module-level function that runs one task in a worker and reads the feed from
    `worker_feed()`. ``setup(*setup_args)`` installs any other per-worker state.
    """

    task: Callable[[Any], Any]
    run: Callable[[Any], Any]
    setup: Callable[..., None] | None = None
    setup_args: tuple = ()
    cache: bool = True


def validate_backend(backend: str) -> None:
    if backend not in BACKENDS:
        raise ValueError("parallel_backend must be one of: serial, thread, process, auto")


def worker_feed() -> ColumnarBarFeed:
    if _FEED is None:
        raise RuntimeError("backtest worker was not initialized")
    return _FEED


def run_cases(
    items: Iterable[Any],
    run_local: Callable[[Any], Any],
    *,
    feed: ColumnarBarFeed,
    backend: str,
    workers: int,
    remaining: int,
    policy: SweepScheduling,
    execution: dict[str, Any],
    process: ProcessPlan | None = None,
    max_chunk: int = 1,
    max_inflight: int | None = None,
) -> Generator[Any, None, None]:
    """Yield ``run_local(item)`` for each item, in input order.

    ``auto`` runs the first item as a pilot, keeps its result, and uses processes
    only when ``policy`` says the ``remaining`` cases repay the startup cost;
    without a ``process`` plan it stays serial. Closing the iterator stops
    submitting work. ``execution`` receives backend, pilot and transport facts.
    """
    items = iter(items)
    pilot_seconds = 0.0
    if backend == "auto":
        execution["backend"] = "serial"
        first = next(items, _NONE)
        if first is _NONE:
            return
        start = perf_counter()
        result = run_local(first)
        pilot_seconds = perf_counter() - start
        execution["pilot_seconds"] = pilot_seconds
        yield result
        backend = policy.backend(pilot_seconds, remaining, workers) if process is not None else "serial"
    chunk_size = policy.chunk_size(pilot_seconds, maximum=max_chunk) if pilot_seconds else 1
    execution.update(backend=backend, chunk_size=chunk_size)
    limit = workers * policy.max_inflight_per_worker
    limit = min(limit, max_inflight) if max_inflight else limit
    if backend == "serial":
        yield from map(run_local, items)
    elif backend == "thread":
        with ThreadPoolExecutor(max_workers=workers) as pool:
            yield from bounded_map(pool, run_local, items, limit=limit)
    else:
        if process is None:
            raise ValueError("process execution requires a ProcessPlan")
        with _shipped_feed(feed, execution) as handle:
            try:
                # spawn, never fork: a forked child inherits Polars' thread pool
                # mid-state and can deadlock (Linux defaults to fork).
                pool = ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn"),
                                           initializer=_initialize_worker,
                                           initargs=(handle, process.cache, process.setup, process.setup_args))
            except (NotImplementedError, PermissionError):
                execution.update(backend="serial", fallback="process startup unavailable")
                yield from map(run_local, items)
                return
            tasks = ((process.run, [process.task(item) for item in chunk]) for chunk in batched(items, chunk_size))
            with pool:
                for batch in bounded_map(pool, _run_chunk, tasks, limit=limit):
                    yield from batch


_NONE = object()


def _run_chunk(task):
    run, chunk = task
    return [run(item) for item in chunk]


@contextmanager
def _shipped_feed(feed: ColumnarBarFeed, execution: dict[str, Any]):
    """What a worker needs to rebuild the feed: Parquet paths, or one IPC file."""
    if type(feed) is ParquetBarFeed:
        # O(files), not O(bars): each worker decodes bounded batches from shared
        # disk with the same filters. Subclasses take the IPC path so custom
        # iteration is honored.
        execution["feed_transport"] = "parquet_paths"
        yield tuple(str(path.resolve()) for path in feed.paths), {
            "batch_rows": feed.batch_rows, "date_col": feed.date_col, "symbols": feed.symbols,
            "start": feed.start, "end": feed.end, "merge_sorted": feed.merge_sorted}
        return
    with TemporaryDirectory(prefix="cortexbacktest-feed-") as directory:
        path = str(Path(directory) / "bars.arrow")
        write_feed_ipc(feed, path)
        execution["feed_transport"] = "ipc"
        yield path, None


def _initialize_worker(handle, cache: bool, setup, setup_args) -> None:
    global _FEED
    path, parquet_options = handle
    _FEED = ParquetBarFeed(path, **parquet_options) if parquet_options is not None else read_feed_ipc(path, cache=cache)
    if setup is not None:
        setup(*setup_args)
