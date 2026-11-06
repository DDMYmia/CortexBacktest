# Public API

The public surface has two tiers.

- **Tier 1, `cortexbacktest.api`.** Twenty-one names, versioned by
  `API_VERSION == "v1"`. Build new integrations on `BacktestConfig`,
  `BrokerConfig` and `run_backtest`. Changing this set is an API version
  decision; `tests/test_api_contract.py` pins it.
- **Tier 2, the extension layer.** `BacktestEngine`, `BrokerSimulator`, feeds,
  matching models, record sinks, and the research, reporting and options
  kernels. They are public and supported, and they change faster than tier 1.

```python
from cortexbacktest.api import BacktestConfig, BrokerConfig, StrategySpec, run_backtest

config = BacktestConfig(
    broker=BrokerConfig(initial_cash=100_000, commission_bps=1),
    metadata={"dataset_version": "vendor-2026-07", "strategy_version": "1.2.0"},
)
result = run_backtest(bars, StrategySpec(MyStrategy, {"fast": 20, "slow": 50}), config=config)
report = result.report()
```

`run_backtest` calls the `StrategySpec` factory once per run, so the factory
must return a new instance. Passing a strategy instance directly also works.

## Contract (tier 1)

- `API_VERSION == "v1"`.
- The default account is a long-only cash account. Market orders fill at the
  **next bar's open**. To fill at the same bar's close, set `trade_on_close=True`
  with a non-open `fill_price`; `trade_on_close=True` with `FillPrice.OPEN` is
  rejected as look-ahead.
- Configurations are frozen dataclasses. Containers are copied shallowly. Each
  run builds a new broker, so cash, positions and orders are isolated between
  runs; nested parameters and custom model objects are not deep-copied.
- A configured run has a `run_manifest` (schema v5): the input data SHA-256, the
  engine and broker configuration, strategy parameters and code identity, and
  caller metadata. `records.write_dir()` also writes `run_manifest.json`.
- Broker records include the fee, slippage, matching and settlement models, cash
  interest and borrow fees, a corporate-action summary, contract multipliers and
  risk rules. The corporate-action summary does not yet include `pay_date`,
  `withholding_tax` or `new_symbol`, so two different configurations can share a
  `run_id`; do not use `run_id` alone as proof that two configurations match.
- The `StrategySpec` factory identity and parameters go into the run manifest.
- `metadata` must be JSON-encodable. A value passed to one call overrides the
  configured value of the same name, including `api_version` and the generated
  `strategy_spec`; metadata is not tamper-proof evidence.
- For high-throughput exploratory sweeps, set `BacktestConfig(build_manifest=False)`
  to skip input and provenance hashing. Keep the default `True` for audits,
  reports and reproducible research.

## Engine lifecycle and provenance

`BacktestEngine` is single-use, including empty or failed runs. Construct a new
engine **and broker** for each parameter case; `optimize_strategy` accepts an
`engine_factory` for this. Reusing a feed is supported.

A direct `BacktestEngine(...)` defaults to `build_manifest=False`. Pass
`build_manifest=True` to hash the input and record provenance. Hashing is linear
in the number of bars; a reused `ColumnarBarFeed` caches its fingerprint. The
CLI opts in with `event-backtest --build-manifest`. Python and platform
descriptions are collected when a manifest is serialized, not during the run,
and are not part of `run_id`.

## Import paths

Import each name from the module in the table below (prefix every module with
`cortexbacktest.`). The package root also re-exports every name for notebook
use; that says nothing about stability. Modules not in the table, such as
`runtime`, `event.loop`, `event.result` and `reporting.result`, are
implementation and have no compatibility promise. `__version__` is at the root.
`tests/test_public_module_paths.py` checks that this table and the root exports
are the same set.

| Canonical module | Public names |
| --- | --- |
| `adapters` | `ExportBundle`, `build_export_dataset`, `build_research_bundle` |
| `api` | `API_VERSION`, `BacktestConfig`, `BrokerConfig`, `CommissionModel`, `RiskRule`, `SlippageModel`, `SnapshotFrequency`, `StrategyFactory`, `StrategySpec`, `run_backtest` |
| `broker` | `BrokerSimulator` |
| `contracts` | `Bar`, `Contract`, `CorporateAction`, `CorporateActionEvent`, `CorporateActionType`, `FactorValue`, `Fill`, `FillPrice`, `ForcedLiquidation`, `MarginCall`, `Order`, `OrderBookEvent`, `OrderBookLevel`, `OrderBookSnapshot`, `OrderRecord`, `OrderStatus`, `OrderType`, `Position`, `Side`, `TargetWeight`, `Tick`, `TimeInForce` |
| `costs` | `BasisPointSlippage`, `CommissionScheme`, `FixedCommission`, `PerShareCommission`, `PercentCommission`, `SquareRootMarketImpact`, `TieredCommission`, `VolumeShareSlippage` |
| `data` | `ContractRoll`, `HistoricalDataStore`, `OrderBookRecorder`, `OrderBookReplayer`, `RollAdjustment`, `SessionWindow`, `TickRecorder`, `TickReplayer`, `TradingCalendar`, `Universe`, `apply_universe_mask`, `build_continuous_futures_bars`, `download_yfinance_bars`, `fetch_ccxt_ohlcv_bars`, `infer_session_hours`, `normalize_ccxt_ohlcv`, `normalize_yfinance_history`, `order_book_snapshot_to_tick`, `write_table` |
| `datasets` | `BacktestDataset`, `BarDiagnostics`, `BarResolution`, `DataQualityReport`, `FactorAvailability`, `DatasetProvenance`, `align_factor_availability`, `bars_to_events`, `diagnose_bars`, `load_backtest_dataset`, `load_backtest_dataset_dir`, `load_target_weights`, `normalize_bars`, `save_backtest_dataset`, `sha256_file`, `validate_bars`, `verify_file_hashes` |
| `engine` | `BacktestEngine`, `BacktestResult`, `ResultRetention` |
| `event.analyzers` | `AnalyzerResult`, `DrawdownAnalyzer`, `EquityCurveAnalyzer`, `EventAnalyzer`, `FillStatsAnalyzer`, `OrderFlowAnalyzer` |
| `event.backtrader_compat` | `BacktraderMigrationNote`, `BacktraderStrategyMixin`, `MIGRATION_NOTES` |
| `event.corporate_actions` | `CorporateActionSchedule` |
| `event.parquet_feed` | `ParquetBarFeed` |
| `event.feed` | `ColumnarBarFeed`, `MarketSlice`, `market_slices_from_bars` |
| `event.financing` | `CashInterestModel`, `FinancingCharge`, `FinancingState`, `SettlementModel`, `ShortBorrowFeeModel` |
| `event.ledger` | `Ledger` |
| `event.manifest` | `RunManifest`, `build_run_manifest` |
| `event.matching` | `BarLimitMatching`, `BidAskSpreadMatching`, `MarketStateMatching`, `MatchingModel` |
| `event.optimization` | `StrategyOptimizationResult`, `optimize_strategy` |
| `event.record_sink` | `EventRecordSink`, `RecordFormat` |
| `event.risk` | `LongOnlyRule`, `MarginRequirementRule`, `MaxGrossExposureRule`, `MaxGroupWeightRule`, `MaxLeverageRule`, `MaxNetExposureRule`, `MaxOrderValueRule`, `MaxPositionValueRule`, `MaxPositionWeightRule`, `RiskEngine` |
| `event.signals` | `crossover`, `crossunder` |
| `event.sizers` | `FixedSizeSizer`, `PercentCashSizer`, `Sizer`, `TargetPercentSizer` |
| `event.strategy_loader` | `load_strategy`, `parse_strategy_kwargs` |
| `event.templates` | `BuyAndHoldStrategy`, `SignalStrategy` |
| `event.timeframes` | `bucket_timestamp`, `replay_bars`, `resample_bars` |
| `event.vnpy_compat` | `BarGenerator`, `CtaTemplate`, `PortfolioStrategyTemplate`, `VnpyBacktestingConfig`, `ticks_to_bars` |
| `execution` | `ExecutionSweepResult`, `run_execution_sweep` |
| `frames` | `factor_matrix`, `forward_returns_matrix`, `price_matrix`, `overnight_returns_matrix`, `returns_matrix`, `stack_factor_and_forward_returns` |
| `pipelines` | `BrokerExecutionConfig`, `ResearchExecutionPipelineConfig`, `ResearchExecutionPipelineResult`, `run_research_execution_pipeline`, `CostScenario`, `RobustnessResult`, `run_robustness_study`, `ExecutionAttribution`, `attribute_execution_gap`, `FuturesRollResult`, `run_futures_roll_backtest` |
| `options` | `OptionAccountView`, `OptionBacktestConfig`, `OptionBacktestResult`, `OptionMarginModel`, `ResearchOptionMargin`, `run_option_backtest`, `OptionContract`, `OptionLifecycleEvent`, `OptionMarketFrame`, `OptionQuote`, `QuotePolicy`, `OptionGreeks`, `black76`, `black_scholes`, `bsm_greeks`, `crr_price`, `implied_volatility`, `MonteCarloPrice`, `asian_price`, `crank_nicolson_price`, `leisen_reimer_price`, `longstaff_schwartz_price`, `monte_carlo_price`, `trinomial_price`, `ScheduledOptionTargets`, `delta_hedge_targets`, `goyal_saretto_signal`, `historical_volatility`, `option_deflated_sharpe`, `probability_of_backtest_overfitting`, `purged_option_split`, `select_option`, `option_frames_from_polars`, `option_scenario_summary`, `run_option_scenarios` |
| `search` | `SearchObservation`, `SearchOptimizer`, `SurrogateOptimizer` |
| `rebalance` | `RebalanceConfig`, `TargetWeightStrategy`, `orders_for_target_weights` |
| `records` | `fills_frame`, `open_orders_frame`, `order_records_frame`, `write_fills_csv`, `write_fills_ndjson`, `write_open_orders_csv`, `write_open_orders_ndjson`, `write_order_records_csv`, `write_order_records_ndjson`, `write_positions_csv`, `write_positions_ndjson` |
| `reporting` | `BenchmarkReport`, `PerformanceReport`, `active_returns`, `align_returns`, `alpha`, `benchmark_report`, `beta`, `event_report`, `event_report_html`, `event_tear_sheet`, `event_tear_sheet_html`, `execution_sweep_report`, `exposure_frame`, `exposure_summary`, `factor_sweep_report`, `group_exposure_frame`, `information_ratio`, `performance_report`, `pnl_reconciliation_frame`, `pnl_reconciliation_summary`, `positions_frame`, `research_report`, `rolling_beta`, `rolling_sharpe`, `trade_stats`, `trade_summary`, `trades_frame`, `turnover_frame`, `turnover_summary`, `write_event_report_html`, `write_event_report_json`, `write_event_tear_sheet_html`, `write_execution_sweep_report_html`, `write_execution_sweep_report_json`, `write_factor_sweep_report_html`, `write_factor_sweep_report_json`, `write_research_report_html`, `write_research_report_json` |
| `research` | `FactorPreprocessConfig`, `FactorResearchReport`, `FactorResearchResult`, `FactorSpec`, `FactorSweepCase`, `FactorSweepResult`, `INDICATORS`, `IndicatorRegistry`, `IndicatorSpec`, `MultipleTestingResult`, `TTestResult`, `WalkForwardResult`, `WalkForwardSplit`, `align_factor_matrices`, `apply_volatility_target`, `aroon_oscillator`, `average_directional_index`, `average_true_range`, `benjamini_hochberg`, `bollinger_band_width`, `bollinger_percent_b`, `cap_weights`, `chaikin_money_flow`, `combine_factor_scores`, `commodity_channel_index`, `cumulative_quantile_returns`, `default_indicator_registry`, `deflated_sharpe_ratio`, `demean_cross_section`, `directional_indicators`, `ema_ratio`, `equal_weight`, `evaluate_factor`, `factor_coverage`, `factor_decay_exponential`, `factor_decay_linear`, `factor_decay_summary`, `factor_information_coefficient`, `factor_long_form`, `factor_rank_autocorrelation`, `factor_stability_report`, `ic_summary`, `inverse_volatility_weights`, `macd_histogram`, `macd`, `momentum`, `money_flow_index`, `neutralize_by_group`, `normalize_weights`, `normalized_average_true_range`, `on_balance_volume`, `percentage_price_oscillator`, `prepare_factor`, `purged_walk_forward_splits`, `quantile_turnover`, `rank_cross_section`, `rolling_volatility`, `rolling_zscore`, `rsi`, `run_factor_research`, `run_factor_sweep`, `run_walk_forward`, `select_top_n`, `simulate_target_weights`, `sma_ratio`, `stochastic_d`, `stochastic_oscillator`, `student_t_two_sided_p_value`, `t_test_mean_zero`, `target_volatility_scale`, `top_bottom_weights`, `vwap_ratio`, `weight_turnover`, `weights_from_signals`, `williams_r`, `winsorize_cross_section`, `zscore_cross_section` |
| `strategy` | `Strategy` |

`engine` also exports `SnapshotFrequency`. The strategy hook protocol in
`event.dispatch` is internal.
