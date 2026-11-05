# Guide

Examples, CLI commands and a feature list. The [README](../README.md) covers
installation and the quickstart.

## Development setup

```bash
git clone <this repo> && cd CortexBacktest
python3.13 -m venv .venv && source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m pytest -q
```

Optional extras: `data` (yfinance, ccxt, pandas), `calendar`
(exchange calendars), `optimizer` (cvxpy), `analytics` (SciPy), `duck`
(DuckDB), `indicators` / `tulip` (external indicator providers), `ml`
(scikit-learn).

## Examples

Every example uses synthetic data and runs offline unless noted:
`python examples/<name>.py`.

| Example | Shows |
| --- | --- |
| `event_driven_demo.py` | A minimal event-driven backtest |
| `order_lifecycle_demo.py`, `stop_orders_demo.py` | Open orders, cancellation, stop and stop-limit orders |
| `partial_fill_demo.py`, `fill_price_models_demo.py`, `market_impact_slippage_demo.py` | Volume-limited fills, fill price models, impact slippage |
| `portfolio_risk_demo.py`, `group_risk_demo.py` | Rejecting orders with portfolio and sector risk rules |
| `corporate_actions_demo.py` | Splits and cash dividends |
| `exposure_report_demo.py`, `benchmark_report_demo.py` | Exposure, turnover and benchmark-relative reports |
| `event_report_export_demo.py`, `report_exports_demo.py`, `showcase_reports.py` | JSON and HTML reports |
| `factor_research_demo.py`, `indicator_factor_demo.py`, `multi_factor_research_demo.py` | Cross-sectional factor research |
| `factor_preprocessing_demo.py`, `factor_sweep_demo.py`, `factor_workflow_demo.py` | Preprocessing, parameter sweeps, the full panel → verdict pass |
| `portfolio_optimization_demo.py` | Optimized weights and a simulated portfolio |
| `rebalance_bridge_demo.py`, `research_to_execution_sweep_demo.py`, `research_execution_pipeline_demo.py` | Target weights through the event engine |
| `dataset_loader_demo.py`, `dataset_persistence_demo.py`, `dataset_event_backtest_demo.py` | `BacktestDataset` loading, saving, and one dataset for both workflows |
| `export_files_demo.py` | Research from plain bar and factor files |
| `yfinance_source_demo.py`, `ccxt_source_demo.py` | Normalizing yfinance and ccxt data (extra `data`) |
| `trustworthy_workflows_demo.py` | Adaptive search with a checkpoint, execution-gap attribution, futures rolls |
| `option_research_demo.py` | A synthetic short-straddle option backtest |
| `spy_short_put_study.py` | A short-put study on real SPY chains (needs a chain store; see [option chain data](option-chain-data-sources.md)) |

## Command line

`cortexbacktest --help` lists every command. The dataset → research → execution
path:

```bash
cortexbacktest dataset-validate --bars bars.csv --factors factors.csv
cortexbacktest dataset-build --bars bars.csv --factors factors.csv --output out/dataset
cortexbacktest dataset-inspect --dataset out/dataset

cortexbacktest factor-research --dataset out/dataset --horizons 1,5,20 \
  --winsorize 0.01,0.99 --rank --zscore \
  --output-json out/research.json --output-html out/research.html

cortexbacktest factor-sweep --dataset out/dataset \
  --case mom_1=momentum:window=1 --case mom_2=momentum:window=2 \
  --horizons 1,5 --output-json out/sweep.json

cortexbacktest execution-sweep --dataset out/dataset \
  --case mom_1=momentum:window=1 --horizons 1,5 --commission-bps 1 \
  --output-json out/execution_sweep.json

cortexbacktest research-execution --dataset out/dataset --horizons 1,5 \
  --top-quantile 0.2 --bottom-quantile 0 --commission-bps 1 --min-trade-value 500 \
  --output-json out/research_execution.json
```

`event-backtest` runs a factor-derived strategy, external target weights
(`--target-weights`) or your own strategy class:

```bash
cortexbacktest event-backtest --dataset out/dataset \
  --strategy my_strategy.py:MyStrategy --strategy-kwargs '{"symbol": "AAPL"}' \
  --fill-price close --no-allow-short --max-position-weight 0.25 \
  --groups-file groups.csv --max-group-weight technology=0.4 \
  --build-manifest --output-json out/event.json --output-html out/event.html
```

Option commands: `option-price` and `option-chain` read a JSON request on stdin;
`option-chain-import` adds daily chain files to a store. See
[options](options-backtesting.md).

## Features

**Event engine.** `Strategy.initialize` / `on_bar` / `on_bars` / `finalize`
callbacks; market, limit, stop and stop-limit orders with expiry, amendment,
cancellation and OCO; fill prices open, close, mid, HLC3, OHLC4 and VWAP;
next-bar open by default; volume-limited partial fills; commission, slippage
and square-root impact models; position, exposure, leverage, group and margin
risk rules; corporate actions; cash interest, borrow fees and settlement delay;
futures multipliers and rolls; streaming Parquet replay; run manifests. See
[execution](execution.md) and [bars and replay](bars-and-replay.md).

**Research.** Indicators with checked inputs; winsorizing, ranking, z-scoring
and group neutralization; IC, rank IC, quantile returns, turnover and decay;
multi-factor scores; sweeps; purged walk-forward splits, deflated Sharpe and
Benjamini–Hochberg control; equal-weight, inverse-volatility, HRP and optimized
portfolios. See [factor research](factor-research-workflow.md) and
[research tools](research-tools.md).

**Reporting.** Drawdowns, period returns, rolling Sharpe and beta, alpha,
information ratio, trades, exposure and turnover; tear sheets; JSON and HTML
reports with inline SVG charts; attribution and risk models. See
[risk and reporting](risk-and-reporting.md).

**Data.** Standard long OHLCV normalization and validation; `BacktestDataset`
directories with a v5 manifest that hashes bars, factors, membership and
corporate actions; point-in-time universes; exchange calendars and session
hours. Tables are Polars; pandas frames are converted once on entry.
