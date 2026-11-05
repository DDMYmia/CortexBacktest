"""Experimental quote-driven vanilla options research API."""
from .backtest import (
    FinancingModel, OptionAccountView, OptionBacktestConfig, OptionBacktestResult,
    OptionMarginModel, ResearchOptionMargin, run_option_backtest,
)
from .chains import (
    CHAIN_COLUMNS, CHAIN_KEY, CHAIN_SCHEMA, SOURCES, ChainFilter, ChainQualityPolicy,
    ChainTimePolicy, OptionChainSource, OptionChainStore, attach_lifecycle, clean_option_chain,
    expiry_events_from_prices, expiry_events_from_underlying, iter_normalized_chunks,
    normalize_option_chain, occ_symbol, option_chain_to_frames, option_contracts_from_chain,
    scan_chain_file, session_closes, validate_option_chain, with_underlying_prices,
)
from .costs import OptionCostModel, ResearchOptionCosts
from .margin import RegTOptionMargin
from .contracts import (
    OptionContract, OptionLifecycleEvent, OptionMarketFrame, OptionQuote, QuotePolicy,
)
from .experiments import (
    option_frames_from_polars, option_scenario_summary, run_option_scenarios,
)
from .pricing import (
    OptionGreeks, black76, black_scholes, bsm_greeks, crr_price, implied_volatility,
)
from .numerical import (
    MonteCarloPrice, asian_price, crank_nicolson_price, leisen_reimer_price,
    longstaff_schwartz_price, monte_carlo_price, trinomial_price,
)
from .research import (
    ScheduledOptionTargets, delta_hedge_targets, goyal_saretto_signal,
    historical_volatility, option_deflated_sharpe, probability_of_backtest_overfitting,
    purged_option_split, select_option,
)

__all__ = [
    "MonteCarloPrice", "asian_price", "crank_nicolson_price", "leisen_reimer_price",
    "longstaff_schwartz_price", "monte_carlo_price", "trinomial_price",
    "CHAIN_COLUMNS", "CHAIN_KEY", "CHAIN_SCHEMA", "SOURCES", "ChainFilter", "ChainQualityPolicy",
    "ChainTimePolicy", "OptionChainSource", "OptionChainStore", "attach_lifecycle",
    "clean_option_chain", "expiry_events_from_prices", "expiry_events_from_underlying",
    "iter_normalized_chunks", "normalize_option_chain", "occ_symbol", "option_chain_to_frames",
    "option_contracts_from_chain", "scan_chain_file", "session_closes", "validate_option_chain",
    "with_underlying_prices",
    "FinancingModel", "OptionAccountView", "OptionBacktestConfig", "OptionBacktestResult",
    "OptionMarginModel",
    "OptionCostModel", "RegTOptionMargin", "ResearchOptionCosts", "ResearchOptionMargin",
    "run_option_backtest", "OptionContract", "OptionLifecycleEvent",
    "OptionMarketFrame", "OptionQuote", "QuotePolicy", "OptionGreeks", "black76",
    "black_scholes", "bsm_greeks", "crr_price", "implied_volatility", "ScheduledOptionTargets",
    "delta_hedge_targets", "goyal_saretto_signal", "historical_volatility",
    "option_deflated_sharpe", "probability_of_backtest_overfitting", "purged_option_split",
    "select_option", "option_frames_from_polars", "option_scenario_summary",
    "run_option_scenarios",
]
