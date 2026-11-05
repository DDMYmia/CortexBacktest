"""Bounded caller-supplied equity execution inputs for named recipe research."""
from datetime import UTC, date, datetime
import math

import polars as pl

from ..strict_json import clock as _clock
from ..contracts import CorporateAction, CorporateActionType
from ..datasets import load_backtest_dataset
from ..event import RebalanceConfig
from ..event.financing import CashInterestModel, ShortBorrowFeeModel
from ..event.risk import MarginRequirementRule
from ..pipelines.research_execution import BrokerExecutionConfig, ResearchExecutionPipelineConfig
from ..pipelines.recipe_execution import execute_recipe_portfolio


def execute_recipe_document(document: dict, portfolio: dict, *, input_sha256: str):
    required = {'contractVersion', 'currency', 'priceBasis', 'asOf', 'receivedAt',
                'bars', 'corporateActions', 'costs'}
    if (set(document) != required or document['contractVersion'] != 'recipe-equity-execution-v1'
            or document['currency'] != 'USD' or document['priceBasis'] != 'raw'):
        raise ValueError('recipe_equity_execution_contract_invalid')
    cutoff, received = _clock(document['asOf']), _clock(document['receivedAt'])
    if not received <= cutoff <= datetime.now(UTC):
        raise ValueError('recipe_execution_receipt_after_cutoff')
    rows = document['bars']
    if not isinstance(rows, list) or not 1 <= len(rows) <= 250_000:
        raise ValueError('recipe_execution_bar_limit')
    bars, openings = [], {}
    for row in rows:
        if set(row) != {'date', 'openAt', 'symbol', 'open', 'high', 'low', 'close', 'volume'}:
            raise ValueError('recipe_execution_bar_fields_invalid')
        close, opening = _clock(row['date']), _clock(row['openAt'])
        if not opening < close <= received or openings.get(close, opening) != opening:
            raise ValueError('recipe_execution_session_clock_invalid')
        openings[close] = opening
        if any(type(row[key]) not in (int, float) or not math.isfinite(row[key])
               for key in ('open', 'high', 'low', 'close', 'volume')):
            raise ValueError('recipe_execution_finite_ohlcv_required')
        bars.append({key: value for key, value in {**row, 'date': close}.items() if key != 'openAt'})
    actions = document['corporateActions']
    if not isinstance(actions, list) or len(actions) > 10_000:
        raise ValueError('recipe_execution_action_limit')
    corporate_actions = []
    for row in actions:
        corporate_actions.append(CorporateAction(**{
            **row, 'action_type': CorporateActionType(row['action_type']),
            'ex_date': date.fromisoformat(row['ex_date']),
            'pay_date': date.fromisoformat(row['pay_date']) if row.get('pay_date') else None}))
    costs = document['costs']
    fields = {'initialCash', 'commissionBps', 'slippageBps', 'cashCreditRate', 'cashDebitRate',
              'shortBorrowAnnualRate', 'initialMargin', 'maxVolumeShare', 'cashBuffer', 'allowFractional'}
    if (not isinstance(costs, dict) or set(costs) != fields
            or type(costs['allowFractional']) is not bool
            or any(type(costs[key]) not in (int, float) or not math.isfinite(costs[key]) or costs[key] < 0
                   for key in fields - {'allowFractional'})
            or not 0 < costs['maxVolumeShare'] <= 1 or not 0 < costs['initialMargin'] <= 1):
        raise ValueError('recipe_explicit_execution_costs_invalid')
    frame = pl.DataFrame(bars)
    # Factor values are unused by the execution bridge. Raw close is never adjusted again.
    dataset = load_backtest_dataset(bars=frame, price_col='close',
        factors=frame.select('date', 'symbol').with_columns(pl.lit(0.).alias('factor')),
        corporate_actions=corporate_actions)
    cfg = ResearchExecutionPipelineConfig(
        rebalance=RebalanceConfig(allow_fractional=costs['allowFractional'], cash_buffer=costs['cashBuffer']),
        broker=BrokerExecutionConfig(initial_cash=costs['initialCash'], commission_bps=costs['commissionBps'],
            slippage_bps=costs['slippageBps'], allow_short=True, max_volume_share=costs['maxVolumeShare'],
            opening_times=openings, cash_interest_model=CashInterestModel(
                credit_rate=costs['cashCreditRate'], debit_rate=costs['cashDebitRate']),
            short_borrow_fee_model=ShortBorrowFeeModel(costs['shortBorrowAnnualRate']),
            risk_rules=(MarginRequirementRule(costs['initialMargin']),)))
    return execute_recipe_portfolio(portfolio, dataset=dataset, config=cfg, input_sha256=input_sha256)
