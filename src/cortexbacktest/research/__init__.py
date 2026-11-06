"""Cross-sectional research subsystem."""

from .term_structure import term_structure as term_structure, curve_roll_schedule as curve_roll_schedule

from .pairs import (
    PairConfig as PairConfig,
    PairModel as PairModel,
    PairFormation as PairFormation,
    PairRecipe as PairRecipe,
    fit_pairs as fit_pairs,
)

from .recipes import (
    RecipeResult as RecipeResult,
    ResidualMomentumConfig as ResidualMomentumConfig,
    residual_momentum as residual_momentum,
    group_budget_weights as group_budget_weights,
)

from .holding import (
    HoldingConfig as HoldingConfig,
    PersistentWeightRecipe as PersistentWeightRecipe,
    apply_holding_recipe as apply_holding_recipe,
)

from .event_study import EventStudyResult as EventStudyResult, event_study as event_study

from .revisions import build_revision_factor as build_revision_factor
from .revisions import RevisionPanel as RevisionPanel

from .diagnostics import (
    cumulative_quantile_returns as cumulative_quantile_returns,
    factor_decay_summary as factor_decay_summary,
    factor_information_coefficient as factor_information_coefficient,
    factor_rank_autocorrelation as factor_rank_autocorrelation,
    ic_summary as ic_summary,
    quantile_turnover as quantile_turnover,
)
from .frames import (
    factor_matrix as factor_matrix,
    forward_returns_matrix as forward_returns_matrix,
    price_matrix as price_matrix,
    overnight_returns_matrix as overnight_returns_matrix,
    returns_matrix as returns_matrix,
    stack_factor_and_forward_returns as stack_factor_and_forward_returns,
)
from .factors import (
    FactorSpec as FactorSpec,
    align_factor_matrices as align_factor_matrices,
    combine_factor_scores as combine_factor_scores,
    factor_decay_exponential as factor_decay_exponential,
    factor_decay_linear as factor_decay_linear,
)
from .indicators import (
    average_true_range as average_true_range,
    bollinger_band_width as bollinger_band_width,
    ema_ratio as ema_ratio,
    factor_long_form as factor_long_form,
    momentum as momentum,
    rolling_volatility as rolling_volatility,
    rolling_zscore as rolling_zscore,
    rsi as rsi,
    sma_ratio as sma_ratio,
    macd as macd,
    stochastic_oscillator as stochastic_oscillator,
    aroon_oscillator as aroon_oscillator,
    average_directional_index as average_directional_index,
    bollinger_percent_b as bollinger_percent_b,
    chaikin_money_flow as chaikin_money_flow,
    commodity_channel_index as commodity_channel_index,
    directional_indicators as directional_indicators,
    macd_histogram as macd_histogram,
    money_flow_index as money_flow_index,
    normalized_average_true_range as normalized_average_true_range,
    on_balance_volume as on_balance_volume,
    percentage_price_oscillator as percentage_price_oscillator,
    stochastic_d as stochastic_d,
    vwap_ratio as vwap_ratio,
    williams_r as williams_r,
)
from .indicator_registry import (
    INDICATORS as INDICATORS,
    IndicatorRegistry as IndicatorRegistry,
    IndicatorSpec as IndicatorSpec,
    default_indicator_registry as default_indicator_registry,
)
from .mining import (
    MiningFactorSpec as MiningFactorSpec,
    auto_mine_factors as auto_mine_factors,
    MiningResult as MiningResult,
    evaluate_holdout as evaluate_holdout,
    frozen_factor_targets as frozen_factor_targets,
    frozen_factor_values as frozen_factor_values,
    mine_factors as mine_factors,
)
from .optimization import (
    hierarchical_risk_parity as hierarchical_risk_parity,
    optimize_weights as optimize_weights,
    apply_volatility_target as apply_volatility_target,
    cap_weights as cap_weights,
    equal_weight as equal_weight,
    inverse_volatility_weights as inverse_volatility_weights,
    normalize_weights as normalize_weights,
    select_top_n as select_top_n,
    target_volatility_scale as target_volatility_scale,
    weights_from_signals as weights_from_signals,
)
from .portfolio import (
    simulate_target_weights as simulate_target_weights,
)
from .preprocessing import (
    FactorPreprocessConfig as FactorPreprocessConfig,
    demean_cross_section as demean_cross_section,
    neutralize_by_group as neutralize_by_group,
    prepare_factor as prepare_factor,
    rank_cross_section as rank_cross_section,
    winsorize_cross_section as winsorize_cross_section,
    zscore_cross_section as zscore_cross_section,
)
from .sweeps import (
    FactorSweepCase as FactorSweepCase,
    FactorSweepResult as FactorSweepResult,
    build_indicator_sweep_cases as build_indicator_sweep_cases,
    factor_stability_report as factor_stability_report,
    run_factor_sweep as run_factor_sweep,
)
from .workflow import (
    FactorResearchReport as FactorResearchReport,
    FactorResearchResult as FactorResearchResult,
    evaluate_factor as evaluate_factor,
    factor_coverage as factor_coverage,
    run_factor_research as run_factor_research,
    top_bottom_weights as top_bottom_weights,
    weight_turnover as weight_turnover,
)
from .validation import (
    DataSnoopingResult as DataSnoopingResult,
    HaircutResult as HaircutResult,
    MultipleTestingResult as MultipleTestingResult,
    TTestResult as TTestResult,
    WalkForwardSplit as WalkForwardSplit,
    WalkForwardResult as WalkForwardResult,
    benjamini_hochberg as benjamini_hochberg,
    deflated_sharpe_ratio as deflated_sharpe_ratio,
    purged_walk_forward_splits as purged_walk_forward_splits,
    reality_check as reality_check,
    run_walk_forward as run_walk_forward,
    sharpe_haircut as sharpe_haircut,
    stationary_bootstrap_indices as stationary_bootstrap_indices,
    student_t_two_sided_p_value as student_t_two_sided_p_value,
    superior_predictive_ability as superior_predictive_ability,
    t_test_mean_zero as t_test_mean_zero,
)

# Keep the public export list statically visible to type checkers.
__all__ = [
    "term_structure", "curve_roll_schedule",
    "PairConfig", "PairModel", "PairFormation", "PairRecipe", "fit_pairs",
    "RecipeResult", "ResidualMomentumConfig", "residual_momentum", "group_budget_weights",
    "HoldingConfig", "PersistentWeightRecipe", "apply_holding_recipe",
    "EventStudyResult", "event_study", "build_revision_factor", "RevisionPanel", "cumulative_quantile_returns", "factor_decay_summary",
    "factor_information_coefficient", "factor_rank_autocorrelation", "ic_summary", "quantile_turnover",
    "factor_matrix", "forward_returns_matrix", "price_matrix", "overnight_returns_matrix", "returns_matrix",
    "stack_factor_and_forward_returns", "FactorSpec", "align_factor_matrices", "combine_factor_scores",
    "factor_decay_exponential", "factor_decay_linear", "average_true_range", "bollinger_band_width",
    "ema_ratio", "factor_long_form", "momentum", "rolling_volatility", "rolling_zscore", "rsi", "sma_ratio",
    "aroon_oscillator",
    "average_directional_index",
    "bollinger_percent_b",
    "chaikin_money_flow",
    "commodity_channel_index",
    "directional_indicators",
    "macd_histogram",
    "money_flow_index",
    "normalized_average_true_range",
    "on_balance_volume",
    "percentage_price_oscillator",
    "stochastic_d",
    "vwap_ratio",
    "williams_r",
    "macd", "stochastic_oscillator", "INDICATORS", "IndicatorRegistry", "IndicatorSpec",
    "default_indicator_registry", "MiningFactorSpec", "auto_mine_factors", "MiningResult", "evaluate_holdout",
    "frozen_factor_targets", "frozen_factor_values", "mine_factors", "apply_volatility_target", "cap_weights",
    "equal_weight", "inverse_volatility_weights", "normalize_weights", "select_top_n",
    "target_volatility_scale", "weights_from_signals", "simulate_target_weights", "FactorPreprocessConfig",
    "demean_cross_section", "neutralize_by_group", "prepare_factor", "rank_cross_section",
    "winsorize_cross_section", "zscore_cross_section", "FactorSweepCase", "FactorSweepResult",
    "build_indicator_sweep_cases", "factor_stability_report", "run_factor_sweep", "FactorResearchReport",
    "FactorResearchResult", "evaluate_factor", "factor_coverage", "run_factor_research", "top_bottom_weights",
    "weight_turnover", "DataSnoopingResult", "HaircutResult", "MultipleTestingResult", "TTestResult",
    "WalkForwardSplit", "WalkForwardResult", "benjamini_hochberg", "deflated_sharpe_ratio",
    "purged_walk_forward_splits", "reality_check", "run_walk_forward", "sharpe_haircut",
    "stationary_bootstrap_indices", "student_t_two_sided_p_value", "superior_predictive_ability",
    "t_test_mean_zero", "hierarchical_risk_parity", "optimize_weights",
]
