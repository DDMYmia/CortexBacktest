# Benchmarks

Recorded numbers are measurements on the machine that produced them, not
guarantees. Raw CSV and JSON are in `benchmark-results/`, indexed by
[its README](../benchmark-results/README.md). Other engines are never
dependencies of CortexBacktest.

## Competitor comparison with P&L reconciliation

`benchmarks/real_data_comparison/` runs every engine in its own interpreter
(`venv_<engine>/`, gitignored) over the same arrays and checks each SMA run's
P&L against a plain ledger of the engine's fill convention (`common.truth_pnl`):
next-open, next-close or same-bar close. Weight-based engines (bt, qstrader)
report P&L without a ledger. A second scenario submits a GTC buy limit at
0.97 x close on every bar to load each engine's open-order book.

The multi-timeframe scenario (`--scenario mtf`, `common.Mtf`) is computed bar by
bar inside each engine's callback: a weekly regime (last completed week above its
10-week mean) gates daily entries on Wilder RSI(14) or the lower Bollinger(20, 2)
band, exits on RSI, the 20-day mean or the regime, and sizes each position from
Wilder ATR(14), so every trade has its own share count. Vectorized and
dataframe engines (vectorbt, freqtrade) run the same code as a timed pre-pass.
Its ledger is `common.mtf_targets` fed to `truth_pnl`.

Engines: CortexBacktest (three conventions), backtrader, backtesting.py,
vectorbt, pybroker, zipline-reloaded, vnpy, lumibot, nautilus_trader, bt,
qstrader, pyalgotrade, cipher-bt, freqtrade, qlib and jesse. freqtrade needs
exchange market metadata, so each equity is relabelled as a real OKX USDT pair;
qlib and jesse read data written by their runners outside the timer.

```bash
python benchmarks/real_data_comparison/prep.py      # deep/wide (Yahoo) and recent (1-minute store)
python benchmarks/real_data_comparison/driver.py --out benchmark-results/competitors-<date>
```

`export_daily.py` builds `recent_daily.parquet` on the data server. The driver
exits non-zero when a P&L differs from its ledger without an entry in
`KNOWN_DIFFERENCES`.

## Recorded results

The current run is `benchmark-results/competitors-2026-11-05/` (252 rows, every
P&L reconciled to its ledger or explained by engine price precision). Headline
numbers, on one Apple-silicon machine:

| Measurement | CortexBacktest | Fastest other engine |
| --- | --- | --- |
| Multi-timeframe strategy, 596 symbols × 2,932 bars | 6.6 s | vectorbt 4.4 s (same 3.8 s strategy pre-pass) |
| SMA crossover, 596 symbols | about 758k bars/s | vectorbt 13.8M bars/s (vectorized); pybroker 205k (event-driven) |
| 19,142 resting GTC limit orders, 34 symbols | 0.29 s | pyalgotrade 1.1 s |

Use the raw files for any claim. Small jobs are often faster serially, because
process start-up and serialization dominate. Retention modes trade memory for
detail: summary retention skips per-order records and keeps aggregate counts.

## CortexBacktest harnesses

```bash
python benchmarks/event_engine_compare.py --bars 10000 --symbols 10 --repeat 3 --scenario all
python benchmarks/event_engine_modes_compare.py --bars 5000 --symbols 10 --scenario multi_order --repeat 3
python benchmarks/event_record_export_compare.py --bars 5000 --symbols 10 --repeat 3
python benchmarks/rebalance_execution_compare.py --dates 500 --symbols 100 --top 10 --repeat 3
python benchmarks/factor_research_compare.py --dates 500 --symbols 500 --repeat 3
python benchmarks/execution_sweep_parallel_compare.py --dates 1200 --symbols 120 --cases 12 --workers 4 --repeat 1 --warmup 0
python benchmarks/steady_state_compare.py --bars 100000
python benchmarks/extreme_scaling_compare.py --bars 100000,1000000
python benchmarks/full_benchmark_suite.py --quick --output-dir benchmark-results/<run-name>
```

`--warmup` defaults to 1. `factor_research_compare.py` also checks rank IC
against Alphalens Reloaded when `alphalens-reloaded` is installed.
`benchmarks/perf_gate.py` measures a fixed set of these against
`benchmarks/perf_baseline.json` (see [CONTRIBUTING](../CONTRIBUTING.md)).

Scenarios of `event_engine_compare.py`:

- `buy_once`: one market order on the first slice.
- `multi_order`: recurring buy and sell orders through the run.
- `order_types`: limit, stop and stop-limit paths.
- `partial_fill`: volume-limited partial fills across bars.
- `open_order_scan`: many long-lived unfilled limit orders across symbols.


## Strategy workloads

`cortexbacktest.strategy_library` holds explicit, reproducible variants of five
common strategy families, used as engineering workloads for the engine. They are
not return claims and not line-by-line copies of any source. The tiers order
strategy structure; a higher tier is not necessarily slower.


| Tier | Class / registry name | Rule | Incremental state |
| --- | --- | --- | --- |
| 1 | `BuyHold` / `buy_hold` | Hold a fixed quantity of every symbol that appears; after a miss or a rejection, retry from the actual position | Target position per symbol |
| 2 | `SmaCross` / `sma_cross` | SMA 15/30; after the slow window warms up, hold while fast > slow, else flat | Two bounded windows per symbol, sliding incremental means |
| 3 | `RsiReversion` / `rsi_reversion` | Wilder RSI(14); buy below 30, exit above 70, keep the target in between. Needs 15 prices for 14 changes; a flat RSI is 50 | Previous close, count, average gain and loss per symbol |
| 4 | `BollingerReversion` / `bollinger_reversion` | 20-bar mean and population standard deviation; buy when the close is below mean − 2σ, exit back at the mean. The window includes the current close | Sliding Welford mean/variance per symbol, recomputed periodically to bound error |
| 5 | `MomentumRotation` / `momentum_rotation` | Every 20 market slices, hold a fixed quantity of the top 10 tradable, warmed-up symbols with a positive 60-bar return | Up to 61 closes per symbol; ranks only on rebalance, top-k heap |

References (accessed 2026-09-09):

- [QuantConnect moving-average cross example](https://github.com/QuantConnect/Lean/blob/master/Algorithm.Python/MovingAverageCrossAlgorithm.py):
  the example uses EMA; this library uses SMA with the same fast/slow trend structure.
- [QuantConnect RSI](https://www.quantconnect.com/docs/v2/writing-algorithms/indicators/supported-indicators/relative-strength-index):
  the indicator definition. The 30/70 thresholds and the holding rule are fixed test parameters.
- [QuantConnect Bollinger Bands](https://www.quantconnect.com/docs/v2/writing-algorithms/indicators/supported-indicators/bollinger-bands):
  the mean and standard-deviation bands; the long mean-reversion rule is this library's.
- [QuantConnect strategy library](https://www.quantconnect.com/docs/v2/writing-algorithms/strategy-library):
  lists momentum rotation; the fixed-quantity top-k version here differs from its long/short ETF allocation.

### Semantics

- Every window counts the bars actually observed for that symbol, not calendar days.
- Indicators, sizing helpers and ledger valuation use `Bar.return_close`: the
  adjusted close when `adj_close` is present. CLOSE fills and the close term of
  HLC3/OHLC4 use it too; OPEN, MID and VWAP use their own inputs. Giving
  `adj_close` does not adjust the other OHLC/VWAP fields, so mixed bases can
  still distort results.
- The default run uses `trade_on_close=False`: fills at the next bar's open.
  Rejections and partial fills follow the real position and the remaining open
  quantity; a reversed target first cancels the unfilled quantity, so exposure is
  not doubled.
- The buy quantity is a parameter. Different prices mean different weights, so
  momentum rotation is not equal-weight.
- Missing symbols are not filled forward. A halted bar does not update indicators
  or place orders. Momentum rotation ranks only symbols present and not halted in
  the slice; a deselected holding that is missing gets its exit once quotes
  resume. Ties break by symbol in reverse order. Delisting settlement and a
  point-in-time universe are the market-data layer's job.

### Usage

```python
from cortexbacktest.api import BacktestConfig, BrokerConfig, StrategySpec, run_backtest
from cortexbacktest.event.parquet_feed import ParquetBarFeed
from cortexbacktest.strategy_library import SmaCross

feed = ParquetBarFeed(["minute/2024-01.parquet", "minute/2024-02.parquet"])
result = run_backtest(
    feed,
    StrategySpec(SmaCross, {"fast": 15, "slow": 30, "quantity": 10}),
    config=BacktestConfig(
        broker=BrokerConfig(initial_cash=1_000_000, commission_bps=1, slippage_bps=1),
        data_validation="stream", build_manifest=True,
        snapshot_frequency="session", result_retention="summary",
    ),
)
```

The dynamic loader also accepts `module:Class` names such as
`cortexbacktest.strategy_library:SmaCross`. The `STRATEGIES` registry can drive
a test matrix; create a new strategy instance for each run.

### Implementation notes

1. **One data pass.** `data_validation="stream"` computes the fingerprint and the
   corporate-action / adjustment conflict checks during replay. The fingerprint
   matches the default `preflight` mode; the manifest's initial cash and strategy
   parameters are still captured before the run. Bad data later in the stream can
   fail the run after earlier events have executed; a failed run returns no
   result. Data files must not change during the run.
2. **Features.** Sliding mean/variance is amortized O(1), RSI recursion is O(1)
   and momentum history is bounded. A separate full-history NumPy reference checks
   signals, fills, net value and prefix causality. No precomputed full-sample
   feature with future data is reused.
3. **Parameter search.** For an exact `ParquetBarFeed`, process workers receive
   the absolute paths, batch size and time column and read batches directly; the
   main process no longer re-encodes the whole input to IPC. Subclasses keep the
   IPC path to respect custom iteration. `max_workers` and the bounded in-flight
   task count apply; there is no RAM-based worker estimate and no cross-worker
   feature cache.
4. **Result retention.** `StreamingStatsAnalyzer` samples once per complete slice
   and keeps online statistics; `requires_records=False` does not force broker
   history retention, while default and custom analyzers still retain it.
   `EquityCurveAnalyzer` stores one point per bar, so a slice with several symbols
   has several points. Record sinks write in batches; Parquet keeps fixed field
   types. A sink must be closed normally for a complete file; there is no crash
   recovery.
5. **Orders and matching.** Expiry checks scan only orders that can expire, and
   amendments keep the original submission order. A global book of 32 orders or
   fewer is still indexed by symbol, so a thousand-symbol replay does not copy
   other symbols' orders. Random order flows are compared with the full-scan path
   on fills, order records and equity.

`StreamingStatsAnalyzer` reports the mean and population variance of returns,
not annualized and not adjusted for external flows, counting a return only when
the previous equity is positive. `DrawdownAnalyzer.max_drawdown_pct` is the
percentage of the largest money drawdown, not the worst percentage drawdown.
Create new analyzers for each run. A final-only snapshot cannot give the run's
Sharpe or CAGR; use online statistics or a full or daily equity curve.

### Running the workloads

```sh
# Five strategies: next-bar open, 1 bp commission and slippage, partial fills,
# manifest, online statistics, Parquet output
.venv/bin/python benchmarks/strategy_workloads.py \
  --output-dir benchmark-results/strategy-suite-local \
  --symbols 1000 --minutes 1000 --repeat 3 --sink

# Expiry candidate set with mixed GTC/DAY orders
.venv/bin/python benchmarks/order_expiration.py

.venv/bin/python -m pytest -q
```

The workloads use seeded synthetic one-minute OHLCV without a real exchange
calendar. Every strategy gets the same input, cash, fees and fill ratio; both
validation modes must produce the same manifest, statistics and final state.
Timing excludes input generation and includes disk reads, validation, hashing,
features, matching, statistics and enabled sinks. The three repeats run in one
process and include warm-cache effects; they are not a confidence interval. The
RSS in the environment JSON is the script's high-water mark, not per strategy.

A synthetic workload does not certify a decade of real minute data. Long-job
checkpoint/recovery, a real point-in-time universe, billion-row scale and
multiple exchange calendars need their own validation.

## Limits

- The numbers describe the recorded machine and Python version. Re-running is a new
  measurement, not a restoration of the old environment.
- A throughput gain for one stage is not an end-to-end speed-up.
- Benchmark output does not prove execution or economic equivalence between engines.
