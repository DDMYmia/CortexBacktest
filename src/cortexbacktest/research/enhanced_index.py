"""Long-only benchmark-relative allocation using an explicitly dated dynamic model."""
import hashlib
import json
import math
import re

import numpy as np
import polars as pl

from ..reporting.dynamic_style_risk import decompose_dynamic_style_risk
from ..reporting.factor_risk import _dated, _numeric


def optimize_enhanced_index(
    estimate: dict, *, benchmark: pl.DataFrame, previous: pl.DataFrame,
    expected_returns: pl.DataFrame, sectors: pl.DataFrame, currency: str,
    expected_return_horizon: str, source_sha256: dict[str, str],
    max_weight: float, max_turnover: float, max_tracking_error: float,
    risk_aversion: float, sector_bounds: dict[str, tuple[float, float]],
    style_bounds: dict[str, tuple[float, float]],
) -> dict:
    """Maximize active expected return less active variance, with explicit hard bounds.

    Tracking error is per model period, never silently annualized. Turnover is
    sum(abs(target-prior)) over assets; residual cash is excluded. Bounds apply to
    target-minus-benchmark sector weights and factor exposures.
    """
    if set(source_sha256) != {'benchmark', 'previous', 'expectedReturns', 'sectors'} or any(
        not isinstance(value, str) or not re.fullmatch(r'[0-9a-f]{64}', value) for value in source_sha256.values()
    ):
        raise ValueError('enhanced_index_source_identities_required')
    if any(type(value) not in (float, int) or not math.isfinite(value) or value < 0
           for value in (max_weight, max_turnover, max_tracking_error, risk_aversion)):
        raise ValueError('enhanced_index_finite_constraints_required')
    if not 0 < max_weight <= 1 or risk_aversion <= 0:
        raise ValueError('enhanced_index_weight_or_risk_aversion_invalid')
    base = {'analysisMode': 'enhanced_index', 'sources': source_sha256,
            'qualificationIssued': False, 'executionAuthorized': False,
            'turnoverBasis': 'target_minus_prior_asset_weight_L1_cash_excluded',
            'riskBasis': 'active_variance_and_tracking_error_per_model_period'}
    # Reuse the existing model label/PSD/specific-risk and benchmark-clock validation.
    checked = decompose_dynamic_style_risk(estimate, benchmark, benchmark, currency=currency,
        portfolio_sha256=source_sha256['benchmark'], benchmark_sha256=source_sha256['benchmark'])
    if checked['status'] != 'available':
        return {**base, 'status': 'unavailable', 'reason': checked['reason'], 'weights': None}
    model = estimate['model']
    if expected_return_horizon != model['horizon']:
        raise ValueError('enhanced_index_expected_return_horizon_mismatch')
    assets = sorted(model['assets'], key=lambda asset: str(asset['instrument_id']))
    symbols = [str(asset['instrument_id']) for asset in assets]
    stamp = benchmark['timestamp'][0]
    benchmark = _dated(benchmark, {'timestamp', 'known_at', 'symbol', 'weight'})
    previous = _dated(previous, {'timestamp', 'known_at', 'symbol', 'weight'})
    alpha = _dated(expected_returns, {'timestamp', 'known_at', 'symbol', 'expected_return'})
    industries = _dated(sectors, {'timestamp', 'known_at', 'symbol', 'sector'})
    for frame in (previous, alpha, industries):
        if frame['symbol'].to_list() != symbols or frame['timestamp'].n_unique() != 1:
            raise ValueError('enhanced_index_complete_aligned_asset_grid_required')
    if (previous['timestamp'][0] > stamp or alpha['timestamp'][0] != stamp or industries['timestamp'][0] != stamp):
        raise ValueError('enhanced_index_input_date_mismatch')
    _numeric(previous, ['weight'])
    _numeric(alpha, ['expected_return'])
    if (industries.schema['sector'] != pl.String or industries['sector'].null_count()
            or any(not name or name != name.strip() for name in industries['sector'])):
        raise ValueError('enhanced_index_sector_labels_required')
    b = benchmark['weight'].to_numpy().astype(float)
    prior = previous['weight'].to_numpy().astype(float)
    mu = alpha['expected_return'].to_numpy().astype(float)
    if np.any(b < 0) or not math.isclose(float(b.sum()), 1., abs_tol=1e-8, rel_tol=0):
        raise ValueError('enhanced_index_fully_invested_long_only_benchmark_required')
    if np.any(prior < 0) or prior.sum() > 1 + 1e-8:
        raise ValueError('enhanced_index_prior_long_only_with_optional_cash_required')
    names = model['factors']
    X = np.array([asset['loadings'] for asset in assets], dtype=float)
    F = np.array(model['covariance'], dtype=float)
    covariance = X @ F @ X.T + np.diag([asset['specific_variance'] for asset in assets])
    covariance = (covariance + covariance.T) / 2
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    scale = max(float(np.abs(covariance).max()), np.finfo(float).tiny)
    if eigenvalues.min() < -scale * 1e-12:
        raise ValueError('enhanced_index_asset_covariance_not_PSD')
    if eigenvalues.min() < 0:
        covariance = (eigenvectors * np.maximum(eigenvalues, 0)) @ eigenvectors.T
    bounds = []
    for mapping, allowed, kind in [(style_bounds, set(names), 'style'),
                                   (sector_bounds, set(industries['sector']), 'sector')]:
        if not isinstance(mapping, dict) or set(mapping) - allowed:
            raise ValueError('enhanced_index_unknown_constraint_label')
        for name, limits in mapping.items():
            if (not isinstance(limits, (list, tuple)) or len(limits) != 2
                    or any(type(value) not in (float, int) or not math.isfinite(value) for value in limits)
                    or limits[0] > limits[1]):
                raise ValueError('enhanced_index_active_bounds_invalid')
            vector = X[:, names.index(name)] if kind == 'style' else np.array(
                [1. if sector == name else 0. for sector in industries['sector']])
            bounds.append((kind, name, vector, float(limits[0]), float(limits[1])))
    configuration = {'maxWeight': max_weight, 'maxTurnover': max_turnover,
        'maxTrackingError': max_tracking_error, 'riskAversion': risk_aversion,
        'sectorBounds': sector_bounds, 'styleBounds': style_bounds, 'currency': currency,
        'horizon': expected_return_horizon}
    base.update(modelSha256=checked['modelSha256'], inputSha256=checked['inputSha256'],
                asOf=stamp.isoformat(), previousAt=previous['timestamp'][0].isoformat(), constraints=configuration)
    if len(symbols) * max_weight < 1 - 1e-10:
        return {**base, 'status': 'infeasible', 'reason': 'weight_caps_cannot_fully_invest', 'weights': None}
    import cvxpy as cp
    from cvxpy.error import SolverError
    w = cp.Variable(len(symbols))
    active = w - b
    variance = cp.quad_form(active, cp.psd_wrap(covariance))
    constraints = [w >= 0, w <= max_weight, cp.sum(w) == 1,
                   cp.norm1(w - prior) <= max_turnover, variance <= max_tracking_error ** 2]
    constraints.extend(expr for _, _, vector, low, high in bounds
                       for expr in (vector @ active >= low, vector @ active <= high))
    objective_scale = max(float(np.abs(mu).max()), risk_aversion * scale, 1e-12)
    problem = cp.Problem(cp.Minimize((risk_aversion * variance - mu @ active) / objective_scale), constraints)
    try:
        problem.solve(solver='CLARABEL')
    except SolverError:
        return {**base, 'status': 'unavailable', 'reason': 'solver_failed', 'weights': None}
    if problem.status != cp.OPTIMAL or w.value is None:
        return {**base, 'status': 'infeasible' if problem.status == cp.INFEASIBLE else 'unavailable',
                'reason': 'solver_' + str(problem.status), 'weights': None}
    result = np.asarray(w.value).ravel()
    tolerance = 1e-7
    def valid(values):
        a = values - b
        return (np.isfinite(values).all() and abs(float(values.sum()) - 1) <= tolerance
                and values.min() >= -tolerance and values.max() <= max_weight + tolerance
                and float(np.abs(values - prior).sum()) <= max_turnover + tolerance
                and float(a @ covariance @ a) <= max_tracking_error ** 2 + tolerance * max(scale, max_tracking_error ** 2)
                and all(low - tolerance <= float(vector @ a) <= high + tolerance
                        for _, _, vector, low, high in bounds))
    if not valid(result):
        return {**base, 'status': 'unavailable', 'reason': 'solver_constraint_violation', 'weights': None}
    result = np.maximum(result, 0)
    result /= result.sum()
    if not valid(result):
        return {**base, 'status': 'unavailable', 'reason': 'normalized_constraint_violation', 'weights': None}
    active_values = result - b
    active_variance = max(0., float(active_values @ covariance @ active_values))
    config_sha = hashlib.sha256(json.dumps(configuration, sort_keys=True, allow_nan=False).encode()).hexdigest()
    return {**base, 'status': 'available', 'configurationSha256': config_sha,
        'weights': dict(zip(symbols, result.tolist(), strict=True)),
        'activeWeights': dict(zip(symbols, active_values.tolist(), strict=True)),
        'trackingError': math.sqrt(active_variance), 'activeVariance': active_variance,
        'activeExpectedReturn': float(mu @ active_values),
        'objective': float(mu @ active_values) - risk_aversion * active_variance,
        'turnover': float(np.abs(result - prior).sum()),
        'activeExposures': [{'kind': kind, 'name': name, 'value': float(vector @ active_values),
                            'lower': low, 'upper': high} for kind, name, vector, low, high in bounds],
        'solver': 'CLARABEL', 'solverStatus': problem.status,
        'historicalInputEvidence': 'caller_supplied_model_and_sources_not_certified'}
