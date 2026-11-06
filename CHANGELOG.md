# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Indicators

- Fourteen built-in indicators: `macd_histogram`, `percentage_price_oscillator`,
  `bollinger_percent_b`, `average_directional_index`, `directional_indicators`,
  `aroon_oscillator`, `stochastic_d`, `williams_r`, `commodity_channel_index`,
  `normalized_average_true_range`, `on_balance_volume`, `money_flow_index`,
  `chaikin_money_flow` and `vwap_ratio`. Tests compare them with stored TA-Lib output.
- `rsi` and `average_true_range` take `smoothing="wilder"`. The default stays the
  simple rolling mean, so existing results do not change.
- `IndicatorRegistry.run_panel` runs an OHLCV indicator for each symbol of a long
  table and returns a factor matrix.
- `stochastic_oscillator` gives null, not NaN, on a flat window.
- `factor-sweep --case` accepts every built-in price indicator, not only five.

### Polars 2.1

- The dependency is now `polars>=2.1,<3` (was `>=1.0,<2`). The `polars-pinned`
  CI job runs the suite on 2.1.0.
- `read_feed_ipc` no longer passes `memory_map=True` to `pl.read_ipc`; Polars 2.x
  removed that argument. Callers are unaffected.
- Run manifests: `run_id` and `engine_code_sha256` hash the engine sources, so
  they change with this release even for identical inputs. Compare runs by
  `data_sha256`, broker and strategy settings, not by `run_id`, across the upgrade.
- Result tables match 1.44.2 bit-for-bit for the event engine. Per-date IC and
  rank-IC values can differ in the last bit (about 1e-17 absolute, 1e-13 relative
  near zero). Tests that compare stored floats need a tolerance. See
  `benchmark-results/polars-2.1-upgrade/`.
- Output fingerprints: `benchmarks/output_fingerprint.py` compares result tables
  bit-exactly or within tolerances; `benchmarks/factor_pipeline_perf.py` times the
  factor pipeline on a realistic-size synthetic panel.

## 0.0.1 — 2026-11-05

First standalone build, distributed as a local wheel. Everything up to the
0.1.0 preview below ships in it.

### Release clean-up

- Process-parallel sweeps and observation-file checks start workers with
  `spawn`. On Linux the default `fork` copied Polars' thread pool into the
  child and the run deadlocked; macOS already used `spawn`.
- Removed ten flat alias modules that no documented import path used:
  `cortexbacktest.analyzers`, `events`, `execution_sweeps`, `factors`,
  `optimization`, `performance_report`, `portfolio`, `preprocessing`, `sweeps`
  and `trades`. Import from the owning subpackage
  (`cortexbacktest.reporting`, `.research`, `.execution`, `.event`), as
  `docs/api.md` lists. The canonical flat paths (`broker`, `costs`,
  `engine`, `frames`, `rebalance`, `records`, `strategy`) stay.
- Removed twelve one-off benchmark scripts superseded by
  `benchmarks/real_data_comparison/`, and the result directories measured before
  the next-bar-open default. `benchmark-results/` keeps the 2026-11-05 competitor
  run, the perf gate and the options research evidence.
- Renamed the file-export adapter: `adapters.fusion_export` is
  `adapters.export_files`, `FusionExportBundle` is `ExportBundle`,
  `build_fusion_dataset` is `build_export_dataset`, and
  `BacktraderMigrationNote.fusion_name` is `cortex_name`.
- Benchmarks no longer import other engines from a `references/` checkout:
  `event_engine_compare.py` and `steady_state_compare.py` time CortexBacktest
  only, `extreme_horizontal_compare.py` is gone, and the competitor comparison
  lives in `benchmarks/real_data_comparison/` alone. `factor_research_compare.py`
  checks rank IC against an installed `alphalens-reloaded`.
- Removed `examples/sentiment_factor_demo.py`; it read the output of a retired
  sibling project.
- Documentation consolidated from 30 public files to 12 and checked against the
  code: `api.md` (was `public-api-v1.md` + `python-imports.md`), `decisions.md`
  (the seven ADRs and the reuse policy), `architecture.md` (with `data-flow.md`),
  `execution.md`, `bars-and-replay.md`, `research-tools.md`,
  `risk-and-reporting.md` and `benchmarks.md`. Historical delivery notes and
  per-run acceptance logs were dropped; they remain in git history.

### Next-open runs skip the idle opening pass

With no open orders, the next-bar-open fill pass only marked each bar, but it
still ran order expiry, `process_bar` and record notification on every bar.
The engine now marks directly when the book is empty (as the same-bar close path
already did), unless the broker overrides `process_bar` or `expire_orders`.
Results are unchanged (equity, fills and notification order are identical to the
full pass). Single-asset synthetic hold: default 426k -> 495k bars/s, throughput
mode 649k -> 869k; 34 real tickers +7.6%; busy books unchanged.

### Standalone build

- Version set to 0.0.1; README rewritten as a standalone package page.
- HTML report titles say CortexBacktest.
- `BacktestConfig(trade_on_close=True)` with the default `BrokerConfig` now raises:
  it used to fill at the decision bar's open. Pass `fill_price=FillPrice.CLOSE`.

#### Default fills at the next bar's open (breaking)

The documented default (`docs/api.md`) was a next-bar open fill, but
`BacktestEngine` defaulted to `trade_on_close=True` and `BrokerSimulator` to
`FillPrice.CLOSE`, so a default run filled on the decision bar's own close.

- `BacktestEngine(trade_on_close=False)`, `BrokerSimulator(fill_price=FillPrice.OPEN)`,
  `run_execution_sweep`, `BacktestDataset.run_event_backtest` and the CLI broker
  options now default to next-bar open.
- An engine-built broker follows the engine's timing: `BacktestEngine(trade_on_close=True)`
  without a broker still fills at the same bar's close.
- `trade_on_close=True` with `FillPrice.OPEN` raises: the decision already saw that
  bar's close, so filling at its open is look-ahead.
- CLI: `--trade-on-open` (event and execution-sweep commands) is replaced by
  `--trade-on-close`; the research pipeline keeps its compatibility flag.
- Migration: to keep the old behaviour pass `trade_on_close=True` and
  `fill_price=FillPrice.CLOSE`. Tests written for the old default now say so.
- The zero-cost market fast path covers open fills too, so the default keeps it.

### PnL reconciliation through corporate actions; equity mining executes

- `BacktestResult.corporate_action_events` carries what each split, dividend,
  delisting, merger and symbol change did to the account (records retained).
- `pnl_reconciliation_summary(..., corporate_actions=...)` and `trades_frame`
  replay those events against the open FIFO lots in time order: a split or stock
  dividend rescales quantity, price and per-unit commission; a delisting or cash
  merger closes the lots at the cash paid; a stock merger carries them to the new
  symbol with the cash leg returned from cost; a symbol change moves them. Cash
  dividends are a new `dividends` line in the summary. The reconciliation error
  was the actions themselves before (500 to 9,000 on a 100k account in the tests;
  ~7k-10k on a real S&P 500 run); it is now zero for every action, long and short.
- `run_mining_job` evaluates US-equity studies on raw prints when the export
  carries `corporate-actions.json` (corporate-actions-v1, digest-bound into the
  result inputs); without it the execution reports `requires_corporate_actions`.

### Minute-data profile; `Bar` is a NamedTuple (breaking)

`benchmarks/minute_partition_profile.py` streams research-layer minute
partitions through `PartitionedBarFeed` and times storage, feed, the engine
loop and a per-bar strategy separately. On 500 tickers x 2 months (8.9M bars)
it found 43% of an idle engine run in one place: the frozen dataclass
`__init__` of `Bar`, which assigns 13 fields through `object.__setattr__`.

- `Bar` is now a `NamedTuple`: still immutable and hashable, built in C.
  Use `copy.replace(bar, ...)` (it works on both the old and the new `Bar`)
  instead of `dataclasses.replace`, and `Bar._fields` / `bar._asdict()`
  instead of `fields` / `asdict`.
- Per bar: feed 1.07 -> 0.39 us; idle engine run 11.4 s -> 5.1 s; engine
  loop itself 0.19 us. A trivial SMA strategy costs 1.6 us per bar, so user
  Python code is now about 74% of a run, the engine loop about 9% and storage
  about 2%. A native event loop would still build a Python `Bar` for every
  `on_bar` call, so it is not worth writing on these numbers.

### Workspace field names (breaking)

One name per fact, as the workspace glossary (`docs/FIELD_NAMES.md` at the
workspace root) and the published contracts spell them:

- Option chains: `option_type` -> `right`, `expiration` -> `expiry`
  (`CHAIN_SCHEMA`, `CHAIN_KEY`, `occ_symbol`). `OptionContract` already said
  `right` and `expiry`, so a chain row and the contract built from it now agree.
- Option rights are `CALL` / `PUT` (as in `order-intent-v2`), not `call` / `put`:
  `OptionRight`, `OptionContract.right`, pricing, margin and research helpers.
  Vendor files are still read in any of their spellings (`C`, `call`, `1`, ...).
- `observed_at` -> `event_time` (as in `factor-observations-v1`):
  `OptionQuote`, `OptionLifecycleEvent`, option-frame columns, and the column
  `align_factor_availability` requires (`FactorAvailability.observed_col` ->
  `event_col`).

### Transaction costs

- Add `OptionCostModel` / `ResearchOptionCosts`: spread, per-contract and
  per-premium commission, ticket minimums, slippage in ticks or basis points,
  and square-root market impact. Defaults reproduce the previous behaviour
  exactly, so no existing result moves; every other component is opt-in.
- O'Donovan and Yu found 17 of 24 delta-hedged sorts significant gross and none
  net of costs, with hedging frequency a first-order term. Hedging turnover is
  therefore not a parameter here: it is whatever the rebalance schedule
  generates, priced one fill at a time.
- A spread sweep now varies the plugin's own `spread_fraction`; with a plugin
  configured, replacing the config field alone would have compared identical
  scenarios. Setting the flat fields alongside a plugin is rejected outright.

### Defined-risk margin

- Add `RegTOptionMargin`: Reg-T style collateral that nets stock-covered short
  options, vertical spreads, iron condors (only the wider wing is charged) and
  same-expiry short combinations. Deterministic pairing, so the number does not
  depend on position ordering.
- The unnetted `ResearchOptionMargin` charges a 390/380 SPY put spread 7,800
  against a real maximum loss of 800 — the protective leg made it cost more than
  the naked short inside it, which put credit spreads, iron condors and covered
  calls out of reach. `RegTOptionMargin` charges 1,000, the strike width.
- `ChainTimePolicy(zone=...)` converts a vendor's time of day per date, so a
  sample spanning daylight-saving changes is not pinned to one UTC hour. The
  Kaggle SPY dataset is a 16:00 America/New_York snapshot over fourteen years,
  where a fixed UTC time is an hour off for half the sample.
- `scripts/fetch_option_chains.py --structural-only` builds a store that keeps
  zero bids, wide spreads and IV outliers: the liquidity filter belongs to the
  study, and a store built at one filter strength cannot be widened later.

### Option-chain ingest performance

- Profile the ingest path (`benchmarks/option_chain_ingest.py`): every Polars
  stage runs at 2.5-25M rows/s and the `OptionMarketFrame` adapter at 411k, so
  the Polars/Python object boundary is the cost, not the cleaning.
- Speed that adapter up 1.79x end to end — sort once and construct positionally
  (1.38x), plus a `utc()` fast path for values already at zero offset (1.17x).
  Both are semantics-preserving and pinned by tests.
- Fuse `normalize_option_chain` into one lazy pass and aggregate expiry closes
  in Polars rather than per chain row.
- Add `iter_normalized_chunks` / `scan_chain_file`: normalise a large vendor
  file a few sessions at a time so a Kaggle year does not have to fit in
  memory. `scripts/fetch_option_chains.py` uses it by default; a test asserts
  the chunked store is identical to the single-shot one.
- Record the GPU finding rather than deferring it: `cudf-polars` needs CUDA,
  the bottleneck is Python object construction that no GPU engine accelerates,
  and the path is kept expression-only so `engine="gpu"` stays a one-line
  change on a machine that has one. See `docs/benchmarks.md`.

### Multiple-testing controls

- Add White (2000) reality check, Hansen (2005) SPA and the Harvey-Liu Sharpe
  haircut to `research.validation`, over a Politis-Romano stationary bootstrap
  that keeps the serial dependence an iid resample would destroy. A family
  padded with weak candidates breaks the reality check and not SPA; that
  difference is pinned by a test.

### Public option-chain ingestion

- Add `cortexbacktest.options.chains`: normalise DoltHub, Kaggle SPY EOD
  and OptionsDX chains into one long schema, separate structural errors from
  known quote defects, filter with per-rule removal counts, store hive-partitioned
  Parquet by ticker and year, and adapt back to contracts and market frames.
- Expiry settlement from the chain's own underlying close is available but
  explicitly an approximation, never the official fixing. See
  `docs/option-chain-data-sources.md` and `scripts/fetch_option_chains.py`.
- Verified against the live sources rather than assumed: the Kaggle SPY EOD
  dataset is OptionsDX's wide bracketed layout, not a long one, and the DoltHub
  API needs the `master` branch, per-day queries (a date range exceeds the
  server's deadline), and `RowLimit` treated as a truncated page. Measured
  coverage limits are recorded in the doc.

### Trustworthy workflows

- Publish resampled bars at bucket completion by default; replay partial bars
  at source availability time. Chart-only start labels remain explicit. Validate
  source interval alignment and handle repeated DST hours separately.
- Add explicit trading-session settlement calendars and market timezones;
  preserve legacy natural-day delays. Calendar contents enter run manifests.
- Add interruption/restart, parallel/cache-mode equivalence, fresh retry and
  out-of-sample isolation regression tests; expose robustness workflows.
- Add opt-in ordered research-to-execution equity attribution with an additive
  reconciliation check, liquidity/risk and weight-constraint effects.
- Add pluggable deterministic adaptive search and a bounded RBF surrogate,
  using the existing case checkpoint and objective interfaces.
- Add a close-first real futures roll workflow with margin, costs, partial-fill
  waiting and explicit incomplete-roll states. See `docs/execution.md`.

### Architecture review remediation

- Reject reused engines, including failed runs, to prevent accumulated state.
- Make direct-engine manifests opt-in; add CLI `--build-manifest`. The
  configuration API retains its provenance default. Defer environment probes.
- Respect instance and subclass broker hooks in both execution modes.
- Declare typed strategy dispatch, separate core result data from reporting,
  and move command orchestration into pipelines.
- Document canonical imports and the 0.3.0 transitional-alias migration in
  `docs/api.md`; preserve existing aliases for 0.1.x and 0.2.x.
