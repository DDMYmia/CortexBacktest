"Frozen covariance VaR hypotheses versus a frozen empirical comparator, evaluated only on future receipts."
from datetime import UTC, date, datetime, time
import hashlib
import json
import math
from statistics import NormalDist
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl

from .risk_calibration import var_backtest
from ..adapters.observation_file import validate_observation_file


def _clock(value):
    stamp = datetime.fromisoformat(value)
    if stamp.utcoffset() is None:
        raise ValueError('aware prospective risk clock required')
    return stamp


def freeze_equity_forecast(report, returns, weights, config, sessions):
    if (not isinstance(config, dict) or set(config) != {'sessions', 'confidence', 'significance'}
        or type(config['sessions']) is not int or not 30 <= config['sessions'] <= 252
        or type(config['confidence']) not in (int, float) or not .8 <= config['confidence'] <= .9999
        or type(config['significance']) not in (int, float) or not .001 <= config['significance'] <= .2):
        raise ValueError('bounded explicit prospective risk policy required')
    built = _clock(report['asOf'])
    closes = sessions['closes']
    if (sessions['calendar'] != 'us_equity' or _clock(sessions['generatedAt']) > built
        or len(closes) != config['sessions'] + 1):
        raise ValueError('complete declared future risk calendar required')
    stamps = [_clock(r['sessionClose']) for r in closes]
    if stamps != sorted(set(stamps)) or stamps[0] <= built:
        raise ValueError('risk forecast must precede every future return interval')
    for row, stamp in zip(closes, stamps, strict=True):
        local = stamp.astimezone(ZoneInfo('America/New_York'))
        if (local.date() != date.fromisoformat(row['sessionDate']) or local.weekday() >= 5
            or local.timetz().replace(tzinfo=None) not in {time(13), time(16)}):
            raise ValueError('declared US closing sessions required')
    ids = sorted(a['instrument_id'] for a in report['model']['assets'])
    columns = [returns.filter(pl.col('symbol') == str(i)).sort('timestamp')['return'].to_numpy() for i in ids]
    panel = np.column_stack(columns)
    snapshot = {r['symbol']: r for r in weights.to_dicts()}
    z = NormalDist().inv_cdf(config['confidence'])
    portfolios = {}
    for role in ('portfolio', 'benchmark', 'active'):
        values = {}
        balance_symbols = ['__CASH__'] + (['__DIVIDEND_RECEIVABLE__'] if '__DIVIDEND_RECEIVABLE__' in snapshot else [])
        for symbol in [*[f'INSTRUMENT:{i}' for i in ids], *balance_symbols]:
            row = snapshot.get(symbol, {'weight': 0., 'benchmark_weight': 0.})
            values[symbol] = (row['weight'] if role == 'portfolio' else row['benchmark_weight'] if role == 'benchmark'
                              else row['weight'] - row['benchmark_weight'])
        realized = panel @ np.array([values[f'INSTRUMENT:{i}'] for i in ids])
        empirical = max(0., -float(np.quantile(realized, 1 - config['confidence'], method='linear')))
        portfolios[role] = {'weights': values, 'modelVar': z * math.sqrt(report['risk'][role]['totalVariance']),
                            'empiricalVar': empirical}
    return {
        'contractVersion': 'equity-risk-forecast-v1', 'frozenAt': built.isoformat(),
        'inputSha256': report['inputSha256'], 'modelSha256': report['modelSha256'], 'currency': 'USD',
        'config': config, 'trainingObservations': returns['timestamp'].n_unique(),
        'portfolios': portfolios,
        'periods': [{'sessionDate': closes[i]['sessionDate'], 'periodStart': stamps[i-1].isoformat(),
                     'timestamp': stamps[i].isoformat()} for i in range(1, len(stamps))],
        'forecastMethod': 'frozen_daily_covariance_gaussian_zero_mean_var_hypothesis',
        'comparisonMethod': 'frozen_empirical_linear_quantile_same_received_training_returns',
        'returnPolicy': ('fixed_frozen_weights_daily_rebalanced_adjusted_close_research_returns_no_fees_cash_zero'
                         + ('_dividend_receivable_zero' if '__DIVIDEND_RECEIVABLE__' in snapshot else '')),
        'calendarEvidence': 'declared_rule_based_US_sessions_not_authenticated_calendar_receipts',
        'status': 'frozen_waiting_for_future_sessions', 'minimumCalibrationIntervals': 30,
        'qualificationIssued': False, 'executionAuthorized': False,
    }


def evaluate_equity_forecast(forecast, prices, *, as_of, through=None):
    cutoff = _clock(as_of)
    frozen = _clock(forecast['frozenAt'])
    if forecast['contractVersion'] != 'equity-risk-forecast-v1' or cutoff < frozen or cutoff > datetime.now(UTC):
        raise ValueError('valid frozen risk forecast and nonfuture cutoff required')
    periods = [r for r in forecast['periods'] if _clock(r['timestamp']) <= cutoff
               and (through is None or r['sessionDate'] <= through)]
    config = forecast['config']
    comparisons = {}
    returns = {}
    if periods:
        document = validate_observation_file(prices, as_of=cutoff)
        if (document['recordVersion'] != 'market-bars-v4' or document['scope']['source'] != 'FMP'
            or document['scope']['adjustment'] != 'total_return'):
            raise ValueError('received future FMP total-return prices required')
        symbols = sorted(k for k in forecast['portfolios']['portfolio']['weights'] if k.startswith('INSTRUMENT:'))
        latest = {}
        for row in document['records']:
            key = 'INSTRUMENT:' + row['instrumentId'], _clock(row['sessionClose'])
            if key not in latest or _clock(row['recordedAt']) > _clock(latest[key]['recordedAt']):
                latest[key] = row
        required = {(s, stamp) for s in symbols for r in periods for stamp in (r['periodStart'], r['timestamp'])}
        if set(latest) != {(s, _clock(stamp)) for s, stamp in required}:
            raise ValueError('exact complete prospective price grid required')
        for role, policy in forecast['portfolios'].items():
            values = []
            for period in periods:
                change = 0.
                for symbol in symbols:
                    a, b = [latest[symbol, _clock(period[k])] for k in ('periodStart', 'timestamp')]
                    if any(r.get('invalidReason') or not r.get('adjustmentProvenance') or r['ohlcBasis'] != 'unadjusted'
                           or type(r['adjClose']) not in (int, float) or not math.isfinite(r['adjClose']) or r['adjClose'] <= 0
                           for r in (a, b)):
                        raise ValueError('latest prospective price revision invalid; no older fallback')
                    change += policy['weights'][symbol] * (b['adjClose'] / a['adjClose'] - 1.)
                values.append(change)
            returns[role] = values
    elif prices is not None:
        raise ValueError('no future price input allowed before first forecast interval closes')
    for role, policy in forecast['portfolios'].items():
        for method, var_key in (('model', 'modelVar'), ('empirical', 'empiricalVar')):
            values = returns.get(role, [])
            result = {'status': 'unavailable', 'reason': 'insufficient_prospective_observations',
                      'totalReturnIntervals': len(values), 'warmupExcluded': 0, 'calibratedIntervals': 0,
                      'warnings': ['fixed_forecast_not_refitted_during_observation_window'], 'summary': None,
                      'observations': [], 'breaches': sum(v < -policy[var_key] for v in values)}
            if len(values) * (1 - config['confidence']) < 5:
                result['warnings'].append('fewer_than_five_expected_calibration_breaches')
            if forecast['trainingObservations'] * (1 - config['confidence']) < 5:
                result['warnings'].append('fewer_than_five_expected_tail_samples_in_training_window')
            if len(values) >= forecast['minimumCalibrationIntervals']:
                observed = pl.DataFrame({'timestamp': [_clock(r['timestamp']) for r in periods],
                                         'period_start': [_clock(r['periodStart']) for r in periods], 'return': values})
                predicted = pl.DataFrame({'timestamp': observed['timestamp'], 'forecast_at': [frozen] * len(values),
                                          'var': [policy[var_key]] * len(values)})
                result.update(var_backtest(observed, predicted, confidence=config['confidence'],
                                            significance=config['significance']).to_dict())
                result.update(status='available', reason=None, calibratedIntervals=len(values))
            comparisons[f'{role}_{method}'] = result
    return {'contractVersion': 'equity-risk-observation-v1',
            'status': 'waiting_for_sessions' if not periods else 'complete' if len(periods) == len(forecast['periods']) else 'partial',
            'forecastSha256': hashlib.sha256(json.dumps(forecast, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest(),
            'frozenAt': forecast['frozenAt'], 'asOf': as_of, 'observedIntervals': len(periods),
            'observationCoverage': {
                'version': 1,
                'dueAtCutoff': sum(_clock(p['timestamp']) <= cutoff for p in forecast['periods']),
                'requestedDue': len(periods), 'acceptedIntervals': len(periods),
                'notYetDue': sum(_clock(p['timestamp']) > cutoff for p in forecast['periods']),
                'excludedByThrough': sum(_clock(p['timestamp']) <= cutoff for p in forecast['periods']) - len(periods),
                'missingPriceIntervals': 0,
                'priceAcceptance': 'exact_complete_grid_required_missing_or_invalid_prices_reject_task',
            },
            'plannedIntervals': len(forecast['periods']), 'config': {'window': forecast['trainingObservations'], **config},
            'portfolios': comparisons, 'qualificationIssued': False, 'executionAuthorized': False}
