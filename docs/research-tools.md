# Research tools

Library functions that sit beside the [factor workflow](factor-research-workflow.md).
Each takes explicit inputs, refuses incomplete data instead of filling it, and
returns results that feed the target-weight bridge and the event engine. None of
them is a trading signal or a claim about returns.

Common rules:

- Clocks are timezone-aware. A `known_at` or `available_at` column says when a
  row could be known; a row known after the decision it feeds is rejected.
- Periods and windows count the rows supplied, not calendar days. The caller
  owns the decision grid and passes explicit nulls for missing observations.
- Missing, non-finite or ambiguous input gives `null` with a reason, never zero.

## Indicators and their inputs

`research.IndicatorRegistry` (`default_indicator_registry()`) runs indicators
with checked inputs:

- `run(name, data, **params)` validates and computes; `assess(...)` returns a
  serializable input report (`usable_rows` per symbol, failure reasons such as
  `insufficient_complete_history`); `describe(...)` returns the requirements.
- Every built-in indicator declares its input shape and warmup. Warmup counts
  the observations including the first usable output: momentum, return
  volatility and RSI need `window + 1` prices; EMA needs `span`; the MACD line
  needs `slow`; rolling price statistics and OHLC kernels need `window`.
  `describe(name, **params)["warmup_bars"]` gives the number for any indicator.
- Price inputs are wide date × symbol matrices; OHLC inputs are single-asset
  tables (`high_col` / `low_col` / `close_col` / `volume_col` aliases are
  honoured). Duplicate or null dates are rejected. A missing or non-finite value
  invalidates every rolling window that contains it.
- `run_panel(name, long_bars, column=None)` runs a table indicator for each
  symbol of a long table and returns a date × symbol factor matrix. A symbol
  without enough complete history gets no column.
- Custom `register` calls may declare required columns, warmup and market or
  frequency scope. A restricted scope needs an explicit `input_context`; it is
  never inferred from symbols.

| Kind | Input | Indicators |
| --- | --- | --- |
| Trend | prices | `momentum`, `sma_ratio`, `ema_ratio`, `macd`, `macd_histogram`, `percentage_price_oscillator` |
| Trend | table | `average_directional_index`, `directional_indicators`, `aroon_oscillator` |
| Oscillator | prices | `rsi`, `rolling_zscore`, `bollinger_percent_b` |
| Oscillator | table | `stochastic_oscillator`, `stochastic_d`, `williams_r`, `commodity_channel_index` |
| Volatility | prices | `rolling_volatility`, `bollinger_band_width` |
| Volatility | table | `average_true_range`, `normalized_average_true_range` |
| Volume | table | `on_balance_volume`, `money_flow_index`, `chaikin_money_flow`, `vwap_ratio` |

`rsi`, `average_true_range` and `normalized_average_true_range` default to a
simple rolling mean. Pass `smoothing="wilder"` for Wilder's recursive average, the
value most charting tools show. Tests compare the indicators with stored TA-Lib
output. RSI, %R, CCI, OBV, MFI, Aroon, %D and %B match on every row. ATR, DI, ADX,
MACD and PPO start from a different seed, so they match only after the recursion
converges: compare them with another library after a long warmup.
`bollinger_percent_b` uses the sample standard deviation, as
`bollinger_band_width` does; TA-Lib uses the population value.

### External providers

`tulip:<name>` (extra `tulip`, `tulipy`) and `pandas_ta:<name>` (extra
`indicators`, `pandas-ta-classic`) are reviewed for `sma`, `ema`, `rsi` and
`atr`. Tulip takes `period`, pandas-ta takes `length`; both default to 20 for
SMA/EMA and 14 for RSI/ATR. Other provider functions and unreviewed options
(`offset`, `fillna`, `talib`, …) are refused rather than run with unknown
timing. A missing provider package raises; there is no fallback to another
implementation.

Each symbol is computed on its contiguous complete segments, so recursive state
(EMA, RSI) restarts after a gap and the declared warmup prefix stays null.

```sh
cortexbacktest factor-sweep --dataset data/ --case trial=tulip:rsi:period=14
```

## Residual momentum, group budgets and pairs

- `research.residual_momentum` fits an OLS of asset returns on factor returns
  over `training_periods` strictly before the scoring periods, skips the most
  recent `skip_periods`, and scores the out-of-fit residual sum divided by its
  standard deviation. `raw` mode needs a risk-free column; `excess` excludes it.
- `research.group_budget_weights` gives each group a fixed gross budget from
  dated membership known at each decision. Budgets never move between groups;
  a missing group leaves cash.
- `research.fit_pairs` freezes `formation_periods` of same-currency prices,
  selects pairs by normalized price distance and fits a log-OLS beta.
  Distance does not establish cointegration; the optional `statsmodels` test is
  diagnostic only. `PairRecipe` adds entry confirmations, cooldown, stop, mean
  crossing and maximum holding. A pair maps to weights `1/(1+β)` and
  `−β/(1+β)`, so β ≠ 1 is not dollar-neutral. Shared legs are netted before
  execution.

The caller supplies borrow availability, FX and corporate-action evidence;
prices alone do not establish them.

## Futures term structure

`research.term_structure` takes a long raw-contract snapshot (`date, root,
symbol, expiry, last_trade_at, quote_at, known_at, price, price_scale,
currency, price_unit, multiplier`). For each snapshot it picks the first two
expiries whose last trade is after the decision and at least
`min_days_to_last_trade` days away, and returns annualized log carry
`(log near − log far) × 365 / expiry gap` (or `near/far − 1` with
`annualized=False`). Stale or asynchronous quotes, future receipts, mixed
currency or unit, and non-positive prices are rejected.

`research.curve_roll_schedule` turns front-contract changes into `ContractRoll`
objects for `pipelines.run_futures_roll_backtest` (see
[execution](execution.md#futures)). Adjusted continuous prices do not replace
a synchronous near/far chain.

## Event study

`research.event_study(returns, market, events, *, as_of, event_window=(-5, 5),
estimation_window=120, estimation_gap=10, model="market")` measures abnormal
returns around declared events.

- `returns` is a wide frame of simple returns, one row per session close;
  `market` has `date` and `market`; `events` has `event_id`, `symbol`,
  `event_at` (nullable) and `available_at`.
- An event anchors at the first return close at or after `available_at`, never
  earlier. Offsets count return rows.
- Models: `market` (OLS), `market_adjusted`, `mean_adjusted`. The estimation
  window (30–500 rows) ends before the event anchor and window.
- Outputs AR, CAR (with parameter-estimation variance), AAR and CAAR. Every
  dropped event has a reason. Overlapping event windows form clusters, and
  cross-sectional t-inference is suppressed for them.

`adapters.news_events.load_news_replay_events` reads a digest-bound news replay file as
events.

## Persistent holdings

`research.HoldingConfig`, `PersistentWeightRecipe` and `apply_holding_recipe`
add entry/exit hysteresis and overlapping cohorts to proposed target weights.

- Enter after `enter_periods` scores at or above `enter_threshold`; exit after
  `exit_periods` scores at or below `exit_threshold` once `min_active_periods`
  have passed; then `cooldown_periods` decisions without entry. A null score
  exits at once and starts the cooldown.
- A basket forms every `formation_interval` decisions and lives
  `holding_periods` decisions. Each basket gets at most
  `max_gross / ceil(holding_periods / formation_interval)`; empty slots stay
  cash. Baskets are summed by symbol before execution.
- `to_document` / `from_document` checkpoint the state as JSON; a restore
  checks it against the pinned config and universe.

These are target-weight controls. They do not guarantee a realized holding
period or a liquidation when the market cannot trade.

## Portfolio allocation

- `research.hierarchical_risk_parity(returns)` takes a complete wide return
  matrix (1–200 symbols), clusters by correlation distance with single linkage
  and splits by inverse variance. Ties break by symbol.
- `research.optimize_weights(...)` solves long-only, fully invested minimum
  variance, or mean variance when `expected_returns` is given, with CLARABEL
  through `cvxpy` (extra `optimizer`). `max_turnover` bounds the L1 change
  against a supplied `previous` vector; it does not count fees or limit executed
  turnover. Only optimal solutions within 1e-7 of every constraint are
  accepted; there is no fallback.

Both return weights that plug into the target-weight frames.

## Model-input diagnostics

`research.model_diagnostics` computes drift checks on model inputs from bars,
without loading a model: population stability (PSI decile bins fitted on the
training window) and rank displacement (needs 50 instruments and 5
observations each, else `null` with the reason).
`research.alpha_diagnostics.alpha_input_diagnostics` recomputes Qlib's
Alpha158 processing (`ProcessInf`, `ZScoreNorm`, `Fillna`) on a bounded feature
snapshot without importing Qlib. `adapters.qlib_predictions.evaluate_qlib_predictions` scores a
`qlib-predictions-v1` file against bars. These are descriptive diagnostics, not
a holdout or a model selection.
