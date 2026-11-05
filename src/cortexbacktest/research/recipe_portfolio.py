"""Explicit rank allocation and persistence for a caller-pinned recipe universe."""
from dataclasses import asdict
from datetime import datetime
import math

import polars as pl

from .holding import HoldingConfig, apply_holding_recipe
from .recipes import RecipeResult, group_budget_weights


def residual_score_portfolio(scores: pl.DataFrame, *, universe_known_at: datetime,
                             top_n: int, long_short: bool, max_gross: float,
                             holding: HoldingConfig) -> dict:
    """Rank supplied scores, then apply holding to selection eligibility, not score sign.

    Short candidates must remain eligible even when their residual scores are negative.
    Missing scores break persistence; deselection emits -1 and selection +1.
    """
    if (not isinstance(universe_known_at, datetime) or universe_known_at.utcoffset() is None
            or not isinstance(holding, HoldingConfig) or holding.max_gross != max_gross
            or holding.enter_threshold != 0 or holding.exit_threshold != -0.01):
        raise ValueError('recipe_explicit_universe_clock_and_selection_holding_required')
    symbols = sorted(set(scores.columns) - {'date'})
    membership = pl.DataFrame([
        {'date': stamp, 'known_at': universe_known_at, 'symbol': symbol, 'group': 'pinned_universe'}
        for stamp in scores['date'].to_list() if stamp >= universe_known_at for symbol in symbols
    ], schema={'date': pl.Datetime('us', 'UTC'), 'known_at': pl.Datetime('us', 'UTC'),
               'symbol': pl.String, 'group': pl.String})
    allocation = group_budget_weights(scores, membership, {'pinned_universe': max_gross},
        top_n=top_n, long_short=long_short, max_gross=max_gross)
    selection = []
    for score, target in zip(scores.sort('date').iter_rows(named=True),
                             allocation.values.iter_rows(named=True), strict=True):
        selection.append({'date': target['date'], **{
            symbol: None if score[symbol] is None or not math.isfinite(score[symbol])
            else 1.0 if target[symbol] != 0 else -1.0 for symbol in symbols}})
    targets, checkpoint = apply_holding_recipe(allocation.values,
        pl.DataFrame(selection, infer_schema_length=None), config=holding)
    output = RecipeResult(targets, allocation.diagnostics, {
        **allocation.summary, 'holding': asdict(holding),
        'universeKnownAt': universe_known_at.isoformat(),
        'universeEvidence': 'caller_pinned_not_historical_membership_certification',
        'holdingSignal': 'selected_plus_one_deselected_minus_one_missing_null',
        'periodBasis': 'provided_decision_rows',
        'output': 'target_weights_not_fills_or_realized_returns'}).to_dict()
    output['checkpoint'] = checkpoint
    return output
