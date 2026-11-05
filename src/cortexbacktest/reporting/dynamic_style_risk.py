"""Period-specific style fits; input evidence is supplied by the frozen-data adapter."""
from datetime import datetime
import math
import re

import numpy as np
import polars as pl

from .risk_model import _model_identity
from .factor_risk import _RESERVED


def _aware(value):
    return isinstance(value, datetime) and value.utcoffset() is not None


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def estimate_dynamic_style_risk(
    returns: pl.DataFrame, exposures: pl.DataFrame, target_exposures: pl.DataFrame, *,
    factors: list[str], method: str, window: int, minimum_observations: int,
    fit_end: datetime, model_known_at: datetime, currency: str, input_sha256: str, model_id: str,
) -> dict:
    """Fit each return cross section from exposures known by its explicit period start.

    Missing periods are retained as unavailable, never dropped to stretch a rolling
    window. WLS_sqrt_cap uses sqrt(cap) loss weights, hence cap**.25 on the design.
    Target exposures are independently dated and are not copied into historical fits.
    """
    _model_identity(model_id, currency, input_sha256)
    if (method not in {'OLS', 'WLS_sqrt_cap'} or type(window) is not int
            or type(minimum_observations) is not int or not 2 <= minimum_observations <= window <= 1260
            or not _aware(fit_end) or not _aware(model_known_at) or fit_end > model_known_at
            or not 1 <= len(factors) <= 7 or len(set(factors)) != len(factors)
            or any(not isinstance(name, str) or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]{0,31}', name)
                   or name in (_RESERVED | {'market', 'period_start', 'market_cap'}) for name in factors)):
        raise ValueError('dynamic_style_configuration_invalid')
    factor_names = ['market', *factors]
    return_fields = {'period_start', 'timestamp', 'symbol', 'known_at', 'return'}
    exposure_fields = {'period_start', 'symbol', 'known_at', *factors}
    if method == 'WLS_sqrt_cap':
        exposure_fields.add('market_cap')
    if (set(returns.columns) != return_fields or set(exposures.columns) != exposure_fields
            or set(target_exposures.columns) != {'timestamp', 'known_at', 'symbol', *factors}
            or not returns.height or returns.height > 126000 or exposures.height > 126000):
        raise ValueError('dynamic_style_input_fields_or_limit')
    for frame, clocks, keys in [
        (returns, ('period_start', 'timestamp', 'known_at'), ('timestamp', 'symbol')),
        (exposures, ('period_start', 'known_at'), ('period_start', 'symbol')),
        (target_exposures, ('timestamp', 'known_at'), ('symbol',)),
    ]:
        if (frame.schema['symbol'] != pl.String or frame['symbol'].null_count()
                or frame.select(*keys).unique().height != frame.height
                or any(not re.fullmatch(r'[1-9][0-9]*', symbol) for symbol in frame['symbol'])):
            raise ValueError('dynamic_style_unique_instrument_ids_required')
        for clock in clocks:
            dtype = frame.schema[clock]
            if not isinstance(dtype, pl.Datetime) or not dtype.time_zone or frame[clock].null_count():
                raise ValueError('dynamic_style_aware_complete_clocks_required')
    lookup = {(row['period_start'], row['symbol']): row for row in exposures.iter_rows(named=True)}
    periods, residuals, fits = [], {}, []
    previous_close = None
    for group in returns.sort('timestamp', 'symbol').partition_by('timestamp', maintain_order=True):
        rows = group.to_dicts()
        close = rows[0]['timestamp']
        starts = {row['period_start'] for row in rows}
        if len(starts) != 1:
            raise ValueError('dynamic_style_shared_period_start_required')
        start = starts.pop()
        if (not start < close <= fit_end or (previous_close is not None and start != previous_close)
                or any(row['known_at'] < close for row in rows)):
            raise ValueError('dynamic_style_nonoverlapping_complete_period_grid_required')
        previous_close = close
        reason = None
        selected = [found for found in (lookup.get((start, row['symbol'])) for row in rows) if found is not None]
        if not len(factor_names) < len(rows) <= 100:
            reason = 'insufficient_cross_section_degrees_of_freedom'
        elif any(row['known_at'] > model_known_at for row in rows):
            reason = 'return_not_received_by_model_cutoff'
        elif any(not _finite(row['return']) or row['return'] < -1 for row in rows):
            reason = 'missing_or_invalid_period_return'
        elif len(selected) != len(rows):
            reason = 'missing_period_exposure'
        elif any(row['known_at'] > start for row in selected):
            reason = 'exposure_not_known_by_period_start'
        elif any(not _finite(row[name]) for row in selected for name in factors):
            reason = 'missing_or_nonfinite_period_exposure'
        elif method == 'WLS_sqrt_cap' and any(not _finite(row['market_cap']) or row['market_cap'] <= 0 for row in selected):
            reason = 'missing_or_invalid_period_market_cap'
        diagnostic = {'periodStart': start.isoformat(), 'timestamp': close.isoformat(),
                      'assets': len(rows), 'degreesOfFreedom': len(rows) - len(factor_names), 'reason': reason}
        if reason is None:
            design = np.array([[1., *[row[name] for name in factors]] for row in selected], dtype=float)
            y = np.array([row['return'] for row in rows], dtype=float)
            scale = np.ones(len(rows))
            if method == 'WLS_sqrt_cap':
                caps = np.array([row['market_cap'] for row in selected], dtype=float)
                scale = (caps / caps.max()) ** .25
            beta, _, rank, singular = np.linalg.lstsq(design * scale[:, None], y * scale, rcond=None)
            if rank != len(factor_names) or singular[0] / singular[-1] > 1e10:
                diagnostic['reason'] = 'rank_deficient_or_ill_conditioned_cross_section'
            elif not np.isfinite(beta).all():
                diagnostic['reason'] = 'nonfinite_period_factor_returns'
            else:
                error = y - design @ beta
                if not np.isfinite(error).all():
                    raise ValueError('dynamic_style_nonfinite_residual')
                fits.append((close, beta))
                residuals[close] = dict(zip([row['symbol'] for row in rows], error.tolist(), strict=True))
                diagnostic['factorReturns'] = dict(zip(factor_names, beta.tolist(), strict=True))
        periods.append(diagnostic)
    if previous_close != fit_end:
        raise ValueError('dynamic_style_fit_end_must_match_last_period')
    fit = {'analysisMode': 'historical_dynamic', 'method': method, 'inputSha256': input_sha256,
           'window': window, 'minimumObservations': minimum_observations,
           'periods': periods, 'factorNames': factor_names, 'specificCovariance': 'diagonal',
           'specificVarianceEstimator': 'sample_variance_of_period_cross_section_residuals_ddof_1',
           'factorCovarianceEstimator': 'sample_covariance_ddof_1', 'annualization': 'none',
           'historicalInputEvidence': 'requires_frozen_source_evidence_not_certified_by_estimator',
           'qualificationIssued': False, 'executionAuthorized': False}
    recent = periods[-window:]
    if len(recent) < minimum_observations or any(row['reason'] is not None for row in recent):
        return {'status': 'unavailable', 'reason': 'incomplete_rolling_factor_window', 'model': None, 'fit': fit}
    selected_dates = {row['timestamp'] for row in recent}
    covariance = np.atleast_2d(np.cov(np.array([beta for stamp, beta in fits if stamp.isoformat() in selected_dates]), rowvar=False, ddof=1))
    assets, missing = [], []
    if not 1 <= target_exposures.height <= 100:
        raise ValueError('dynamic_style_target_asset_count_invalid')
    for row in target_exposures.sort('symbol').iter_rows(named=True):
        if row['timestamp'] != model_known_at or row['known_at'] > model_known_at:
            raise ValueError('dynamic_style_target_exposure_clock_invalid')
        if any(not _finite(row[name]) for name in factors):
            return {'status': 'unavailable', 'reason': 'missing_target_style_exposure',
                    'missingInstruments': [row['symbol']], 'model': None, 'fit': fit}
        samples = [residuals[datetime.fromisoformat(period['timestamp'])].get(row['symbol']) for period in recent]
        if any(value is None for value in samples):
            missing.append(row['symbol'])
            continue
        specific = float(np.var(samples, ddof=1))
        if not math.isfinite(specific):
            raise ValueError('dynamic_style_nonfinite_specific_variance')
        assets.append({'instrument_id': int(row['symbol']), 'loadings': [1., *[row[name] for name in factors]],
                       'specific_variance': specific})
    if missing or not np.isfinite(covariance).all():
        return {'status': 'unavailable', 'reason': 'incomplete_target_specific_history',
                'missingInstruments': missing, 'model': None, 'fit': fit}
    return {'status': 'available', 'model': {'model_id': model_id, 'currency': currency,
            'horizon': 'one_period', 'analysis_mode': 'historical_dynamic',
            'fit_end': fit_end.isoformat(), 'model_known_at': model_known_at.isoformat(),
            'factors': factor_names, 'covariance': covariance.tolist(), 'assets': assets}, 'fit': fit}


def decompose_dynamic_style_risk(
    estimate: dict, portfolio: pl.DataFrame, benchmark: pl.DataFrame, *,
    currency: str, portfolio_sha256: str, benchmark_sha256: str,
) -> dict:
    """Consume an estimated dynamic model through the existing Euler-risk engine.

    Both portfolios must provide every model asset (including explicit zero weights)
    on the model date. Active risk uses actual portfolio minus benchmark weights.
    """
    import hashlib
    import json
    from .factor_risk import _dated, _numeric, factor_risk_decomposition

    for digest in (portfolio_sha256, benchmark_sha256):
        if not isinstance(digest, str) or not re.fullmatch(r'[0-9a-f]{64}', digest):
            raise ValueError('dynamic_risk_portfolio_source_digest_required')
    fit = estimate.get('fit', {})
    if (fit.get('analysisMode') != 'historical_dynamic'
            or fit.get('qualificationIssued') is not False or fit.get('executionAuthorized') is not False):
        raise ValueError('dynamic_risk_estimate_contract_required')
    identity = {'analysisMode': 'historical_dynamic', 'inputSha256': fit['inputSha256'],
                'portfolioSourceSha256': portfolio_sha256, 'benchmarkSourceSha256': benchmark_sha256,
                'qualificationIssued': False, 'executionAuthorized': False}
    if estimate.get('status') == 'unavailable' and estimate.get('model') is None:
        return {**identity, 'status': 'unavailable', 'reason': estimate.get('reason'),
                'missingInstruments': estimate.get('missingInstruments', []), 'fit': fit}
    if estimate.get('status') != 'available':
        raise ValueError('dynamic_risk_estimate_status_invalid')
    model = estimate['model']
    if model['analysis_mode'] != 'historical_dynamic' or currency != model['currency']:
        raise ValueError('dynamic_risk_model_currency_or_mode_mismatch')
    stamp = datetime.fromisoformat(model['model_known_at'])
    end = datetime.fromisoformat(model['fit_end'])
    names = model['factors']
    assets = model['assets']
    if (not isinstance(names, list) or not names or len(names) != len(set(names))
            or not isinstance(assets, list) or not assets
            or any(len(asset['loadings']) != len(names) for asset in assets)):
        raise ValueError('dynamic_risk_model_dimensions_invalid')
    symbols = sorted(str(asset['instrument_id']) for asset in assets)
    if len(symbols) != len(set(symbols)):
        raise ValueError('dynamic_risk_duplicate_model_asset')
    weights = []
    for frame in (portfolio, benchmark):
        frame = _dated(frame, {'timestamp', 'known_at', 'symbol', 'weight'})
        _numeric(frame, ['weight'])
        if frame['symbol'].to_list() != symbols or frame.filter(pl.col('timestamp') != stamp).height:
            raise ValueError('dynamic_risk_complete_model_date_weights_required')
        weights.append(frame)
    p, b = weights
    active = p.with_columns(
        (p['weight'] - b['weight']).alias('weight'),
        pl.max_horizontal(p['known_at'], b['known_at']).alias('known_at'))
    exposures = pl.DataFrame([{'timestamp': stamp, 'known_at': stamp, 'symbol': str(asset['instrument_id']),
                               **dict(zip(names, asset['loadings'], strict=True))} for asset in assets])
    specific = pl.DataFrame([{'timestamp': stamp, 'known_at': stamp, 'symbol': str(asset['instrument_id']),
                             'variance': asset['specific_variance']} for asset in assets])
    matrix = model['covariance']
    if len(matrix) != len(names) or any(len(row) != len(names) for row in matrix):
        raise ValueError('dynamic_risk_covariance_dimensions_invalid')
    covariance = pl.DataFrame({'factor': names, **{name: [row[i] for row in matrix]
                                                  for i, name in enumerate(names)}})
    reports = {label: factor_risk_decomposition(frame, exposures, covariance, specific,
        as_of=stamp, model_known_at=stamp, fit_end=end, model_id=model['model_id'],
        currency=currency, horizon=model['horizon'])
        for label, frame in [('portfolio', p), ('benchmark', b), ('active', active)]}
    return {**identity, 'status': 'available', 'modelSha256': hashlib.sha256(
        json.dumps(model, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest(),
        'modelId': model['model_id'], 'modelKnownAt': stamp.isoformat(), 'currency': currency,
        'horizon': model['horizon'], 'reports': reports,
        'weightTotals': {'portfolio': float(p['weight'].sum()), 'benchmark': float(b['weight'].sum()),
                         'active': float(active['weight'].sum())},
        'cashRisk': 'excluded_from_asset_weights_assumed_zero',
        'sourceEvidence': 'caller_bound_digests_not_independent_source_certification'}
