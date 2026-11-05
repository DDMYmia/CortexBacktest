"""Event-driven backtesting subsystem."""


from .corporate_actions import (
    CorporateActionSchedule as CorporateActionSchedule,
)
from .broker import (
    BrokerSimulator as BrokerSimulator,
    BrokerState as BrokerState,
)
from .analyzers import (
    AnalyzerResult as AnalyzerResult,
    DrawdownAnalyzer as DrawdownAnalyzer,
    EquityCurveAnalyzer as EquityCurveAnalyzer,
    EventAnalyzer as EventAnalyzer,
    FillStatsAnalyzer as FillStatsAnalyzer,
    OrderFlowAnalyzer as OrderFlowAnalyzer,
)
from .backtrader_compat import (
    BacktraderMigrationNote as BacktraderMigrationNote,
    BacktraderStrategyMixin as BacktraderStrategyMixin,
    MIGRATION_NOTES as MIGRATION_NOTES,
)
from .costs import (
    BasisPointSlippage as BasisPointSlippage,
    CommissionModel as CommissionModel,
    CommissionScheme as CommissionScheme,
    FixedCommission as FixedCommission,
    PercentCommission as PercentCommission,
    PerShareCommission as PerShareCommission,
    SlippageModel as SlippageModel,
    SquareRootMarketImpact as SquareRootMarketImpact,
    TieredCommission as TieredCommission,
    VolumeShareSlippage as VolumeShareSlippage,
)
from .engine import (
    BacktestEngine as BacktestEngine,
    BacktestResult as BacktestResult,
    ResultRetention as ResultRetention,
)
from .events import (
    FillEvent as FillEvent,
    MarketEvent as MarketEvent,
    OrderEvent as OrderEvent,
)
from .execution import (
    ExecutionHandler as ExecutionHandler,
)
from .feed import (
    ColumnarBarFeed as ColumnarBarFeed,
    MarketSlice as MarketSlice,
    market_slices_from_bars as market_slices_from_bars,
)
from .parquet_feed import (
    ParquetBarFeed as ParquetBarFeed,
)
from .financing import (
    CashInterestModel as CashInterestModel,
    FinancingCharge as FinancingCharge,
    FinancingState as FinancingState,
    SettlementModel as SettlementModel,
    ShortBorrowFeeModel as ShortBorrowFeeModel,
)
from .ledger import (
    AccountState as AccountState,
    Ledger as Ledger,
)
from .matching import (
    BarLimitMatching as BarLimitMatching,
    BidAskSpreadMatching as BidAskSpreadMatching,
    MarketStateMatching as MarketStateMatching,
    MatchingModel as MatchingModel,
)
from .manifest import (
    RunManifest as RunManifest,
    build_run_manifest as build_run_manifest,
)
from .optimization import (
    StrategyOptimizationResult as StrategyOptimizationResult,
    optimize_strategy as optimize_strategy,
)
from .orders import (
    OrderManager as OrderManager,
)
from .record_sink import (
    EventRecordSink as EventRecordSink,
    RecordFormat as RecordFormat,
)
from .rebalance import (
    RebalanceConfig as RebalanceConfig,
    TargetWeightStrategy as TargetWeightStrategy,
    orders_for_target_weights as orders_for_target_weights,
    quantity_changes_for_target_weights as quantity_changes_for_target_weights,
)
from .risk import (
    LongOnlyRule as LongOnlyRule,
    MaxGroupWeightRule as MaxGroupWeightRule,
    MaxGrossExposureRule as MaxGrossExposureRule,
    MaxLeverageRule as MaxLeverageRule,
    MarginRequirementRule as MarginRequirementRule,
    MaxNetExposureRule as MaxNetExposureRule,
    MaxOrderValueRule as MaxOrderValueRule,
    MaxPositionValueRule as MaxPositionValueRule,
    MaxPositionWeightRule as MaxPositionWeightRule,
    RiskEngine as RiskEngine,
)
from .sizers import (
    FixedSizeSizer as FixedSizeSizer,
    PercentCashSizer as PercentCashSizer,
    Sizer as Sizer,
    TargetPercentSizer as TargetPercentSizer,
)
from .signals import (
    crossover as crossover,
    crossunder as crossunder,
)
from .strategy import (
    Strategy as Strategy,
)
from .strategy_loader import (
    load_strategy as load_strategy,
    parse_strategy_kwargs as parse_strategy_kwargs,
)
from .templates import (
    BuyAndHoldStrategy as BuyAndHoldStrategy,
    SignalStrategy as SignalStrategy,
)
from .timeframes import (
    bucket_timestamp as bucket_timestamp,
    replay_bars as replay_bars,
    resample_bars as resample_bars,
)
from .vnpy_compat import (
    BarGenerator as BarGenerator,
    CtaTemplate as CtaTemplate,
    PortfolioStrategyTemplate as PortfolioStrategyTemplate,
    VnpyBacktestingConfig as VnpyBacktestingConfig,
    ticks_to_bars as ticks_to_bars,
)

# Keep the public export list statically visible to type checkers.
__all__ = [
    "CorporateActionSchedule", "BrokerSimulator", "BrokerState", "AnalyzerResult", "DrawdownAnalyzer",
    "EquityCurveAnalyzer", "EventAnalyzer", "FillStatsAnalyzer", "OrderFlowAnalyzer",
    "BacktraderMigrationNote", "BacktraderStrategyMixin", "MIGRATION_NOTES", "BasisPointSlippage",
    "CommissionModel", "CommissionScheme", "FixedCommission", "PercentCommission", "PerShareCommission",
    "SlippageModel", "SquareRootMarketImpact", "TieredCommission", "VolumeShareSlippage", "BacktestEngine",
    "BacktestResult", "ResultRetention", "FillEvent", "MarketEvent", "OrderEvent", "ExecutionHandler",
    "ColumnarBarFeed", "MarketSlice", "market_slices_from_bars", "ParquetBarFeed", "CashInterestModel",
    "FinancingCharge", "FinancingState", "SettlementModel", "ShortBorrowFeeModel", "AccountState", "Ledger",
    "BarLimitMatching", "BidAskSpreadMatching", "MarketStateMatching", "MatchingModel", "RunManifest",
    "build_run_manifest", "StrategyOptimizationResult", "optimize_strategy", "OrderManager",
    "EventRecordSink", "RecordFormat", "RebalanceConfig", "TargetWeightStrategy", "orders_for_target_weights",
    "quantity_changes_for_target_weights", "LongOnlyRule", "MaxGroupWeightRule", "MaxGrossExposureRule",
    "MaxLeverageRule", "MarginRequirementRule", "MaxNetExposureRule", "MaxOrderValueRule",
    "MaxPositionValueRule", "MaxPositionWeightRule", "RiskEngine", "FixedSizeSizer", "PercentCashSizer",
    "Sizer", "TargetPercentSizer", "crossover", "crossunder", "Strategy", "load_strategy",
    "parse_strategy_kwargs", "BuyAndHoldStrategy", "SignalStrategy", "bucket_timestamp", "replay_bars",
    "resample_bars", "BarGenerator", "CtaTemplate", "PortfolioStrategyTemplate", "VnpyBacktestingConfig",
    "ticks_to_bars",
]
