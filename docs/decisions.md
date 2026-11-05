# Decisions

Each decision states what holds today. A change to one of these needs a new
entry here and the tests that enforce it.

## 1. Two workflows, one set of contracts (2026-06-20)

Cross-sectional research (`research/`, matrix-first) and event-driven
backtesting (`event/`, an ordered event loop) are peers; neither wraps the
other. They share `contracts.py`, record encodings and reporting. Target weights
are the bridge: research produces them, `rebalance` turns them into orders. See
[architecture](architecture.md).

## 2. Scope: bars and factors in, simulated execution and evaluated results out (2026-07-28)

**In scope** — anything that decides whether a result is correct given its
inputs:

- execution semantics: order lifecycle, matching, partial fills, volume caps,
  price limits, halts, costs and slippage;
- position and portfolio accounting, including corporate actions that change a
  held position (splits, cash and stock dividends, delistings, symbol changes,
  mergers);
- factor evaluation and the guards against self-deception: purged and
  walk-forward splits, multiple-testing control;
- reproducibility: manifests, seeds, provenance of what ran;
- the input *contract* (schema, validation, quality checks) and adapters thin
  enough to satisfy it;
- performance and memory of all of the above.

**Out of scope** — the caller's job:

- data acquisition, warehousing, versioning, lineage and refresh (input hash
  checks stay in scope);
- point-in-time fundamental databases and their revision histories;
- reference data beyond what execution needs (FX conversion, cross-exchange
  session catalogues);
- experiment registries, dashboards and research workflow management;
- live trading and order routing;
- a product UI.

Where an out-of-scope capability is needed to evaluate a strategy, the engine
defines and validates an input contract and lets the caller supply the data.
New feature proposals state which side of this line they fall on.

## 3. Polars, PyArrow and NumPy; no pandas in the core (2026-06-22)

Core tables are Polars, persistence is Parquet through PyArrow, and NumPy is
used where direct array code is clearer. Event contracts are plain Python
objects. pandas is only an optional boundary for data-source adapters and
explicit `backend="pandas"` calls; `scripts/clean_install_smoke.py` checks that
a runtime import does not pull in pandas, SciPy or matplotlib. Python is
`>=3.13,<3.14`.

## 4. No native extension yet (2026-07-05, reviewed 2026-09-05)

Speed work stays in Python, Polars and NumPy: columnar market slices, broker
fast paths, batched record sinks, memory-mapped worker input and process
parallelism. Rust/PyO3 is reconsidered only when all of these hold:

- profiles at realistic size show a stable, isolated hotspot in the event loop
  or broker;
- it survives indexing, batching, snapshot reduction, streaming records,
  vectorization and process parallelism;
- the native boundary keeps the public Python strategy and contract APIs;
- a measured gain on representative workloads justifies the build and CI cost.

`benchmarks/perf_gate.py` guards the measured workloads; see
[benchmarks](benchmarks.md).

## 5. Runtime composition and import boundaries (2026-09-05)

- Engines are single-use, including after an empty or failed run. Each run
  needs a new engine and broker.
- A direct `BacktestEngine` builds no manifest unless asked
  (`build_manifest=True`); `api.run_backtest` builds one by default.
- `event/loop.py` and `event/result.py` do not import reporting.
  `reporting/result.py` adds report views to the pure result, and `runtime.py`
  composes the public engine from the two.
- `record_rows.py` owns record schemas and row encodings shared by sinks and
  reports.
- `event/dispatch.py` binds optional strategy hooks once per run.
- `cli.py` parses and prints; `pipelines/commands.py` orchestrates.
- One canonical import path per public name, listed in [the API](api.md).

## 6. Reusing other engines (2026-06-20)

The code is MIT. Architecture ideas from any engine may be reused and
reimplemented from scratch. Source code may be copied only when its license
allows it in an MIT project (for example Apache-2.0, with its notice) or when a
recorded authorization covers it; GPL, AGPL and Commons-Clause code
(backtrader, backtesting.py, lumibot, vectorbt) is not copied into
`src/cortexbacktest`. Every adapted piece of code gets an entry in
[THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md) and tests. Other engines
are never runtime dependencies; they appear only in the benchmark harness, each
in its own environment.
