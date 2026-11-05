"""Bridge research targets to the existing event engine."""
from datetime import datetime
import math
import polars as pl

def execute_recipe_portfolio(portfolio: dict, *, dataset, config, input_sha256: str):
    """Send declared targets to the existing event bridge with explicit financing.

    The dataset supplies raw execution bars and corporate actions. This entry does
    not certify caller data, borrow availability, or historical membership.
    """
    from ..contracts import FillPrice
    from ..event.financing import ShortBorrowFeeModel
    from ..event.risk import MarginRequirementRule
    from ..pipelines.research_execution import ResearchExecutionPipelineConfig, _execute_targets
    import re

    if (not isinstance(config, ResearchExecutionPipelineConfig)
            or not isinstance(input_sha256, str) or not re.fullmatch(r'[0-9a-f]{64}', input_sha256)
            or portfolio.get('qualificationIssued') is not False
            or portfolio.get('executionAuthorized') is not False
            or portfolio['summary'].get('output') != 'target_weights_not_fills_or_realized_returns'):
        raise ValueError('recipe_target_execution_binding_required')
    targets = pl.DataFrame([{**row, 'date': datetime.fromisoformat(row['date'])}
                            for row in portfolio['values']], infer_schema_length=None)
    symbols = sorted(set(targets.columns) - {'date'})
    if not targets.height or not symbols:
        raise ValueError('recipe_execution_targets_empty')
    short = any(value < 0 for symbol in symbols for value in targets[symbol])
    broker = config.broker
    if short and (not broker.allow_short or not isinstance(broker.short_borrow_fee_model, ShortBorrowFeeModel)
                  or not any(isinstance(rule, MarginRequirementRule) for rule in broker.risk_rules)):
        raise ValueError('recipe_short_requires_explicit_borrow_fee_and_margin')
    if not broker.opening_times or broker.trade_on_close or broker.fill_price != FillPrice.OPEN:
        raise ValueError('recipe_explicit_session_openings_required')
    for value in (broker.initial_cash, broker.commission_bps, broker.slippage_bps):
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError('recipe_execution_capital_or_cost_invalid')
    for rule in broker.risk_rules:
        if isinstance(rule, MarginRequirementRule) and (
                not math.isfinite(rule.initial_margin) or rule.initial_margin <= 0):
            raise ValueError('recipe_margin_requirement_invalid')
    if broker.initial_cash <= 0:
        raise ValueError('recipe_execution_positive_capital_required')
    if broker.short_borrow_fee_model is not None:
        model = broker.short_borrow_fee_model
        if (not math.isfinite(model.annual_rate) or model.annual_rate < 0
                or type(model.days_per_year) is not int or model.days_per_year <= 0):
            raise ValueError('recipe_borrow_rate_invalid')
    # No fallback from absent price labels to another instrument or unfilled zero exposure.
    bars = dataset.bars
    if set(symbols) != set(bars['symbol']):
        raise ValueError('recipe_execution_asset_grid_mismatch')
    if set(bars['date']) != set(broker.opening_times):
        raise ValueError('recipe_execution_session_grid_mismatch')
    return _execute_targets(targets, dataset.to_event_feed(), dataset=dataset, cfg=config,
        decision_times=True, run_metadata={'scope': 'recipe_historical_diagnostic',
            'recipeInputSha256': input_sha256, 'qualificationIssued': False,
            'executionAuthorized': False, 'borrowAvailability': 'not_certified',
            'financingBasis': 'caller_declared_model'})
