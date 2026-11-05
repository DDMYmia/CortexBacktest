# Contributing to CortexBacktest

This is a 0.x preview. The engine core and the result API work and are covered
by tests, but the public API is not frozen.

## Setup

```bash
python3.13 -m venv .venv && source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m pytest -q
```

Python 3.13 is required (`requires-python = ">=3.13,<3.14"`), and CI tests only
that version.

## Before opening a pull request

Run what CI runs:

```bash
python -m ruff check src tests
python -m basedpyright src/cortexbacktest   # must stay at 0 errors
python -m compileall -q src tests
python -m pytest -q
```

For anything touching packaging, dependencies, or imports, also run the
clean-install gate — it is the only check that catches an undeclared
dependency, because a dev environment usually has the optional extras installed:

```bash
python -m build --sdist
python -m venv /tmp/cleanenv
/tmp/cleanenv/bin/python -m pip install dist/*.tar.gz
cd /tmp && /tmp/cleanenv/bin/python "$OLDPWD/scripts/clean_install_smoke.py"
```

## Before changing anything on the event/execution hot path

Run the performance gate and compare it to the committed baseline:

```bash
python benchmarks/perf_gate.py --output benchmark-results/perf-gate/latest.json
python benchmarks/compare_perf_gate.py \
    --baseline benchmarks/perf_baseline.json \
    --current benchmark-results/perf-gate/latest.json
```

It measures open-order scan throughput, throughput and peak memory at 100k and
1M bars, serial vs. process execution-sweep throughput, and Polars feed
construction — the four dimensions a change to `event/`, `execution/`, or
`data/history.py` is most likely to move without anyone noticing until the
engine is slow at scale. 25% tolerance; a regression beyond that prints
`REGRESSION` and exits non-zero.

Nothing runs this automatically yet, so it is the actual gate right now, not a
nice-to-have: run it yourself before committing a hot-path change, the same
way you'd run pytest. `.github/workflows/perf-gate.yml` runs the same check on
every PR (soft-launch: it warns rather than blocks — see that file) whenever
this repo has somewhere to push to.

### Option-chain ingest

The ingest path has its own harness, because none of the five gate dimensions
touch it:

```bash
python benchmarks/option_chain_ingest.py --rows 400000 --frame-rows 200000 \
    --output benchmark-results/option-chain-ingest/latest.json
```

It reports per-stage throughput and the peak-RSS increase each stage
contributes. It is deliberately *not* wired into `perf_gate.py`: the committed
`perf_baseline.json` is machine-specific, and adding metrics to it would
invalidate the existing baseline for everyone rather than only for the person
who regenerates it. Run this one by hand when changing `options/chains.py`,
`options/experiments.py`, or `options/contracts.py`, and compare a run before
and after the change on the same machine.

Once you're confident a change is a real, intended improvement (not just
today's machine being quiet), refresh the baseline:

```bash
cp benchmark-results/perf-gate/latest.json benchmarks/perf_baseline.json
```

## What to know before changing things

- **Polars only in the core.** The engine and reporting layers are Polars-first
  by decision ([`docs/decisions.md`](docs/decisions.md)).
  pandas is an optional extra confined to the data-source boundary in
  `src/cortexbacktest/data/`. Do not import pandas, scipy, or matplotlib
  from a runtime path — `scripts/clean_install_smoke.py` asserts their absence.
- **Other engines stay out of the dependencies.** The competitor comparison in
  `benchmarks/real_data_comparison/` runs each engine in its own `venv_<engine>/`
  (gitignored). A test that needs another engine must skip when it is not
  installed (`pytest.importorskip`). Ideas may be borrowed; code may not be
  copied into `src/` without recording the basis in
  [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
- **Correctness claims need reproduction.** Performance and parity numbers in
  `docs/` come from `benchmarks/` scripts with raw output under
  `benchmark-results/`. If you change a number in a doc, regenerate the data.
- **Numbers over adjectives.** This engine is compared against mature projects;
  claims like "faster" belong in a table with a method note, not in prose.

## Commits and pull requests

Keep unrelated changes in separate commits. Fill in the pull request template,
including the verification checkboxes.

## Reporting problems

Use the issue templates. Security issues go through
[`SECURITY.md`](SECURITY.md) instead of the public tracker.
