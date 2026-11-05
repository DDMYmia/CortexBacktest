# Architecture

CortexBacktest has two peer workflows that share one vocabulary:

- **Cross-sectional research** (`research/`): date × symbol prices and factors
  in; IC, quantile returns, turnover and target weights out. Matrix code.
- **Event-driven backtesting** (`event/`): bars and strategy callbacks in;
  orders, fills, positions, cash and an equity curve out. An ordered event loop.

Target weights are the bridge between them (`rebalance`). The decisions behind
this layout are in [decisions](decisions.md).

## Layers

```text
                        ┌──────────────────────────────┐
   orchestration        │  pipelines/   cli.py         │  research → execution flows, CLI commands
                        └──────────────┬───────────────┘
                                       │
        ┌──────────────┬───────────────┼───────────────┬──────────────┐
        ▼              ▼               ▼               ▼              ▼
   ┌─────────┐   ┌──────────┐   ┌────────────┐   ┌──────────┐  ┌───────────┐
   │research/│   │ event/   │   │ datasets/  │   │reporting/│  │execution/ │
   │ factors │   │ engine   │   │ load/check │   │ reports  │  │ sweeps    │
   └────┬────┘   └────┬─────┘   └─────┬──────┘   └────┬─────┘  └─────┬─────┘
        │             │               │               │              │
        └─────────────┴───────┬───────┴───────────────┴──────────────┘
                              ▼
                     ┌──────────────────┐
   utilities         │ data/            │  DataFrame backend, calendar,
                     │                  │  universe, history, sources, futures
                     └────────┬─────────┘
                              ▼
                     ┌──────────────────┐
   vocabulary        │ contracts.py     │  Bar / Order / Fill / Position /
                     │                  │  CorporateAction, shared by all layers
                     └──────────────────┘

   options/ is its own kernel: contracts → options/{chains,pricing,margin,backtest}.
   It borrows settlement and records from event/ and does not use the bar feed.
   adapters/ read files in published contracts (observation files, prediction
   files, account reports) into these layers; they hold no engine state.
```

| Package | Owns |
| --- | --- |
| `contracts.py` | `Bar`, `Order`, `Fill`, `Position`, `CorporateAction` and the enums |
| `event/` | Loop, feeds, broker, ledger, matching, costs, financing, risk rules, sizers, record sinks, optimization |
| `research/` | Indicators, preprocessing, factor evaluation, sweeps, walk-forward and multiple-testing guards, portfolio construction |
| `datasets/` | `BacktestDataset`: normalization, quality checks, persistence with a v5 manifest |
| `data/` | Calendars and session hours, point-in-time universes, partitioned history, data sources, continuous futures |
| `reporting/` | Performance, trades, exposure, attribution, risk models, JSON/HTML reports and charts |
| `execution/`, `pipelines/` | Running research output through the event engine; end-to-end commands |
| `options/` | Option pricing, chains, quote-driven option backtests |
| `api.py` | The tier-1 configuration API (see [API](api.md)) |

## Dependency rules

Enforced by `tests/test_architecture_boundaries.py`:

- `research/` and `event/` do not import each other.
- `event/loop.py`, `event/result.py`, `event/record_sink.py` and
  `event/dispatch.py` do not import reporting. `reporting/result.py` adds report
  views to the pure result, and `runtime.py` composes the public engine.
- No module imports a consuming application's backend.
- The flat modules (`broker`, `costs`, `engine`, `frames`, `rebalance`,
  `records`, `strategy`) are one-line re-exports and nothing else.

By convention:

- `pipelines/` orchestrates the layers but holds no business logic of its own.
- Adapters normalize input; they own no strategy, broker or portfolio behaviour.
- Reporting reads results and never mutates engine state.

`data ⇄ event` looks like a cycle but is not: `event` uses the DataFrame helpers
and calendar in `data/` (the calendar imports are `TYPE_CHECKING` only), while
`data/bar_stream.py` builds the event layer's `ColumnarBarFeed`. So `data/`
holds utilities below `event` and one adapter above it.

## Data in

| Entry | Output | Contract |
| --- | --- | --- |
| `data.download_yfinance_bars`, `data.fetch_ccxt_ohlcv_bars` | Long bar table | `date, symbol, open, high, low, close, volume, adj_close` |
| `datasets.load_backtest_dataset` | `BacktestDataset` | Daily by default (`resolution="session"`); intraday needs a calendar with session hours |
| `event.feed.ColumnarBarFeed.from_polars` / `from_bars` | Event feed | `Bar.timestamp` is the time the bar becomes available |
| `event.parquet_feed.ParquetBarFeed` | Streaming event feed | Time-ordered Parquet files, read in batches |
| `data.HistoricalDataStore` | Partitioned history | Hive-partitioned Parquet, one partition at a time |

See [bars and replay](bars-and-replay.md) for the daily/intraday contract.

## Data out

```text
BacktestResult ─┬─ .performance.equity_curve()  → Polars frame
                ├─ .report()                    → JSON-encodable dict
                ├─ .records.write_dir()         → Parquet/CSV/NDJSON + run_manifest.json
                └─ reporting.write_report_html(…, charts=equity_curve_charts(curve))
                                                → one HTML file, inline SVG, no script
```

The run manifest (schema v5) carries the input SHA-256, engine and broker
configuration, strategy identity and caller metadata. A saved dataset has its own
v5 manifest that hashes bars, factors, membership and corporate actions.

## Invariants

- No look-ahead by default: market orders fill at the next bar's open.
- Research outputs target weights; the event engine executes orders.
- A broken example in `examples/` means the API changed.
- Speed work follows profiling (decision 4).
