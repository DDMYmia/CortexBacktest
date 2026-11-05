# Execution

How orders become fills, and the tools that check a result is trustworthy.
Everything on this page is opt-in except the default fill timing (next bar's
open) and immediate settlement.

## Time semantics

`resample_bars(bars, "5m", source_timeframe="1m")` labels each output bar with
its **period end**: the five one-minute bars of 09:30–09:34 give a complete OHLC
to the strategy at 09:35. Input bars are labelled by their start; an explicit
`source_timeframe` checks that source periods align and do not cross a target
boundary. A short tail period aggregates the observations it has and is still
labelled at the period end; the caller decides whether the data is complete.

`timestamp_label="start"` gives start labels for charts; do not use them as the
time a strategy may see the bar. Default bucketing follows the wall clock and does
not infer which session an overnight bar belongs to or where the exchange closes.
Prepare the source by session first, or pass a calendar with session hours so a
bucket that crosses the close ends at the close (see
[bar resolution and session hours](bars-and-replay.md)). Daily
and weekly periods end at local midnight; intraday lengths are real elapsed time.

`replay_bars(..., source_timeframe="1m")` labels each forming higher-timeframe
bar at the time each source bar becomes available. Without `source_timeframe`,
the source timestamp already is the availability time. Replayed values are
cumulative: a cumulative volume is not new volume.

A bar's timestamp is when the engine dispatches the observation. Custom feeds,
precomputed indicators and `on_before_bars` strategies keep their own
availability contract; the engine does not audit arbitrary Python, and it does not
turn a completed bar's historical open into a new tradable quote.

## Matching, costs and financing

- `BarLimitMatching` (default) triggers limit and stop orders on the bar's OHLC
  and caps fills at a share of bar volume; the rest stays open.
- `BidAskSpreadMatching` prices market and limit orders off a synthetic spread;
  other order types use bar matching.
- `MarketStateMatching` wraps another model: orders stay open on halted bars,
  buys cannot fill on a bar locked at limit-up, sells cannot fill at limit-down,
  and matched prices outside the declared daily limits are rejected. The check
  is on the price before slippage, so slippage can still push a fill past the
  limit.
- Commission and slippage models are in `cortexbacktest.costs`.
- `CashInterestModel` and `ShortBorrowFeeModel` charge financing; each charge is
  a `FinancingCharge` record. `SettlementModel(delay_days=...)` keeps pending
  cash apart from settled cash.

A run consumes its engine even when it fails. Strategy finalization, analyzer
completion and sink closure run on the success path only; an exception does not
roll back fills already applied. Retry with a new engine.

### Trading-day settlement

```python
from cortexbacktest.broker import BrokerSimulator
from cortexbacktest.data import TradingCalendar
from cortexbacktest.event.financing import SettlementModel

calendar = TradingCalendar.from_sessions(["2026-09-04", "2026-09-08", "2026-09-09"])
model = SettlementModel.trading_days(calendar, delay_days=2, timezone="America/New_York")
broker = BrokerSimulator(settlement_model=model)
```

The caller excludes weekends and holidays explicitly. Trading-day settlement
counts market-local dates, keeps the local clock time and handles DST shifts. With
a market time zone, timestamps must be aware; a non-session date or a calendar
that does not cover the date is rejected, not extrapolated. Calendar dates live in
the immutable model and go into the run manifest. The model is not an exchange's
fixed daily settlement time, and it does not infer the session of overnight trades.

`SettlementModel(delay_days=2)` is a calendar-day delay. Trading-day mode needs
an explicit calendar.

## Futures

`data.build_continuous_futures_bars` builds one continuous symbol from a
contiguous `ContractRoll` schedule, with no adjustment, additive (difference)
or multiplicative (ratio) back adjustment, roll gaps inferred from the old and
new closes or given explicitly, and volume and open interest preserved. Use it
for signals, not for P&L.

The ledger handles contract multipliers, variation-margin settlement, realized
P&L on closed quantity and a blended basis when adding. `MarginRequirementRule`
checks projected exposure before commission. Round-trip `trades_frame` P&L
omits multipliers; use `pnl_reconciliation_summary` to reconcile fills.

### Rolling real contracts

```python
from cortexbacktest.pipelines import run_futures_roll_backtest

result = run_futures_roll_backtest(
    raw_contract_bars, rolls,
    quantity=2, initial_cash=100_000,
    contract_multipliers={"F1": 10, "F2": 10},
    initial_margin=0.1, commission_bps=1,
)
```

The input is unadjusted per-contract bars and a declared `ContractRoll` chain
with at least one roll; completing a roll needs entry data before the first roll.
Empty or early-ending data can still return `scheduled`. The runner keeps a fixed
signed contract count (negative is short). It closes the old contract for real
and opens the new one on a later bar only after the close fully fills. Each leg
uses the open price, volume limits, halt and price-limit models, margin checks,
commission and slippage, with futures mark-to-market accounting. Different
multipliers can change notional exposure; this is not a notional-preserving roll.

An adjusted continuous-contract spread is never booked as trading P&L. Partial
fills wait; missing quotes keep `waiting_close` / `waiting_open`; orders not fully
done show `closing` / `opening`; rejections and expiries raise. The initial order
is cancelled at the first roll date and only the real position rolls. A position
already flat before a roll is marked `skipped_flat`.

Check `result.rolls` statuses together with the fills, positions and equity in
`result.event`; returning does not mean every roll completed. The default manifest
includes the roll schedule and rules. This is a standalone fixed-contract roll
runner, not a transparent roll plugin for any strategy, and it does not simulate
atomic calendar-spread orders, delivery or options.

## Research-to-execution gap

Set `explain_execution=True` in `ResearchExecutionPipelineConfig` and read
`result.attribution.to_dict()` or `result.report()["execution_attribution"]`.
It is off by default; on, it adds five control backtests.

The decomposition, in order:

1. Research net value × initial cash → zero cost, no volume limit, fractional
   quantity, zero-lag close execution: `target_weight_model_gap`.
2. The real execution time and price convention, one-session signal lag:
   `execution_timing`.
3. Cash buffer, minimum trade value and whole quantities: `weight_constraints`.
4. Real volume limits and risk rules: `unfilled_and_risk`.
5. Slippage: `slippage`.
6. Commission, giving the actual result: `commission`.

Each item is the final-equity difference between neighbouring scenarios, positive
for a gain, and the items sum to actual minus research equity.
`reconciliation_residual` checks only that arithmetic; it does not prove the stage
labels or causality. The split depends on the order and includes position and cost
interactions. Costs already charged in research go to the first item; commission
and slippage records are diagnostics and are not added twice. Unfilled orders and
risk are one item, with no per-order opportunity cost. The decomposition needs
equity snapshots, a deterministic strategy and risk rules (copied between
scenarios). The zero-lag reference is not evidence that a live strategy can trade it.

## Pluggable adaptive search

```python
from cortexbacktest.event.optimization import optimize_strategy
from cortexbacktest.search import SurrogateOptimizer

result = optimize_strategy(
    bars, MyStrategy,
    method="adaptive", max_tries=30, random_state=7,
    optimizer=SurrogateOptimizer(warmup=5, candidate_pool=512),
    checkpoint_path="search.sqlite",
    fast=[5, 10, 15, 20], slow=[30, 50, 100],
)
```

The built-in model is an RBF-kernel surrogate that picks the next parameters by
predicted score plus an uncertainty bonus; numbers are range-scaled and categories
one-hot encoded. The candidate pool and the fitted history are bounded. The
default budget is at most 50 runs; the search ends at the budget, when a plugin
returns `None`, or on `early_stop`. Under strict constraints a finite pool can run
out of candidates early; the search does not guarantee an exhaustive or global
optimum. Configuration values must be JSON-encodable, and adaptive mode needs a
finite objective.

The extension point is the `SearchOptimizer` protocol: a JSON-encodable
`configuration` property and `suggest(grid, observations, *, random_state,
constraint)`. Observations hold parameters and scores; the plugin returns an
unvisited valid parameter set or `None`. State is rebuilt from the full ordered
history, so a fixed seed and configuration give deterministic suggestions.
Duplicate, out-of-range and constraint-violating suggestions are rejected. The
plugin class, source and configuration are part of the checkpoint identity, so a
changed configuration never reuses old results.

Adaptive search depends on the previous result and runs serially; asking for
several thread or process workers raises. Grid and random search still run in
parallel. Without a seed, adaptive search uses 0. The synthetic single-peak tests
prove the feedback loop and the saved cases, not live performance.

Tests cover the guarantees: completed cases are not rerun after an
interruption; a change in data, parameters, code version or seed rejects an old
checkpoint; thread, process, cached and low-memory feeds match the serial
result; a retry uses a new engine and strategy; changing a fold's out-of-sample
prices cannot change its training selection, and each fold starts from new cash
and a cold strategy.

### Robustness study

`from cortexbacktest.pipelines import CostScenario, run_robustness_study` splits
non-overlapping OOS windows by `timestamp.date()` and does not infer exchange
sessions; later folds can use earlier history. Checkpoints belong to
`optimize_strategy`; the robustness study does not resume.

## Offline example

```sh
.venv/bin/python examples/trustworthy_workflows_demo.py --output /tmp/cortex-workflows
```

It writes the adaptive-search SQLite checkpoint, out-of-sample research
Parquet/JSON, and a `summary.json` with the execution gap and roll results.
Running the same command again shows completed search cases being resumed.
