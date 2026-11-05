# Polars 1.44.2 vs 2.1.0: performance and output comparison

Measured on one Linux x86_64 sandbox (Python 3.13.18), same code except the one
edit the upgrade needs: `event/feed_storage.py` calls `pl.read_ipc` without
`memory_map=True`, because Polars 2.x removed that argument. Both interpreters
run the same workloads, interleaved, in separate processes.

## Factor research pipeline

`benchmarks/factor_pipeline_perf.py --symbols 1500 --dates 2520` (about 10 years of
business days, 1500 symbols, 4 factors, horizons 1/5/20, 5 quantiles, 5 bps cost),
one warm-up and three timed repeats.

| | Polars 1.44.2 | Polars 2.1.0 | ratio |
| --- | ---: | ---: | ---: |
| median total, 4 factors | 52.0 s | 52.1 s | 1.00 |
| repeat spread | 51.8–52.7 s | 51.5–52.4 s | |
| cells per second | 291,017 | 290,160 | 1.00 |
| peak RSS | 2,002 MB | 1,946 MB | 0.97 |

Per-factor times, 1.44 vs 2.1, differ by less than 1.5 s on a 13 s factor, which
is within the repeat spread. The out-of-core default in 2.x was not exercised:
this panel fits in memory.

## Output fingerprint

`benchmarks/output_fingerprint.py` writes every result table as canonical JSON,
floats as `float.hex()`, so a match means the same IEEE-754 values. The Polars
thread count is pinned to 1 inside the script; with multiple threads the same
build changed the last bit of some results between two runs.

Factor research (300 symbols, 800 days, four factors) and a `SmaCross` event
backtest with commissions and slippage: 70 tables.

- Same build, two runs (1.44 vs 1.44): 70 of 70 bit-identical.
- Polars 1.44.2 vs 2.1.0: 43 of 70 bit-identical, including every event-engine
  table (equity, fills, orders, summary). The other 27 are per-date IC and rank-IC
  tables and their summaries. Largest absolute difference 5.6e-17, largest relative
  difference 1.1e-13 on values that are close to zero. All 27 pass with
  `--abs-tol 1e-15 --rel-tol 1e-12`. See `fingerprint-compare-tables.txt`.
- Run-manifest differences are limited to `engine_code_sha256` and `run_id`, which
  hash the engine source (the one-line edit changes them by design). These are
  excluded from the comparison; the remaining manifest fields match.

## Repository examples

All 34 examples in `examples/`, run with `PYTHONHASHSEED=0` and one Polars thread,
their stdout, exit codes and written files compared across the two versions
(`examples-comparison.txt`).

- Exit codes and stdout: identical for all 34.
- Written files: identical except two kinds, both checked by content:
  - `trust/*.parquet`: the bytes differ only in the `created_by` metadata string
    (Polars version); the data frames are equal.
  - `trust/adaptive.sqlite`: differs in the `updated_at` wall-clock column, and two
    runs of the same build also differ there. Every other column matches.
- `factor_workflow_demo` fails on both versions, because `data/panel` is not
  produced by any script in this repository (see the review notes). Only the
  traceback paths differ.

## Not covered

- Factor workloads larger than memory (where the 2.x out-of-core default matters).
- Options, execution-sweep and parallel-process outputs beyond the test suite.
- Polars 2.0.0 by itself; only 2.1.0 was compared.
