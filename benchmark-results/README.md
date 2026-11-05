# Benchmark results

One directory per measured run, with its own README and raw files. A recorded
number describes the machine and Python version that produced it; re-running is
a new measurement.

- [competitors-2026-11-05](competitors-2026-11-05/): CortexBacktest against fifteen
  other engines on real daily data (SMA, multi-timeframe, resting orders), every
  P&L reconciled to a ledger of the engine's fill convention.
- [perf-gate](perf-gate/): the CI performance gate's latest measurement, compared
  against `benchmarks/perf_baseline.json`.
- [options-research-2026-09-06](options-research-2026-09-06/): options research
  plumbing evidence on synthetic quotes.

Earlier runs (July–October 2026) measured the engine before the next-bar-open
default and were removed on 2026-11-05; they remain in the Git history.
