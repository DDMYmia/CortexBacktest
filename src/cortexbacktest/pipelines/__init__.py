"""End-to-end workflow orchestration."""

from .research_execution import (
    BrokerExecutionConfig,
    ResearchExecutionPipelineConfig,
    ResearchExecutionPipelineResult,
    run_research_execution_pipeline,
)
from .robustness import CostScenario, RobustnessResult, run_robustness_study
from .attribution import ExecutionAttribution, attribute_execution_gap
from .futures_roll import FuturesRollResult, run_futures_roll_backtest
from .mining_execution import frozen_factor_quantity_changes, run_frozen_factor_simulation, run_mining_execution_pipeline
from .factor_strategy import FactorValuation, FrozenFactorStrategy

__all__ = [
    "FrozenFactorStrategy",
    "FactorValuation",
    "BrokerExecutionConfig",
    "ResearchExecutionPipelineConfig",
    "ResearchExecutionPipelineResult",
    "run_research_execution_pipeline",
    "CostScenario",
    "RobustnessResult",
    "run_robustness_study",
    "ExecutionAttribution",
    "attribute_execution_gap",
    "FuturesRollResult",
    "run_futures_roll_backtest",
    "run_mining_execution_pipeline",
    "run_frozen_factor_simulation",
    "frozen_factor_quantity_changes",
]
