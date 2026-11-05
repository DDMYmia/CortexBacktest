"""Frozen US monthly factors consumed without asserting historical vintage availability."""
from __future__ import annotations

import calendar
import csv
import hashlib
import io
import math
import re
import zipfile
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import polars as pl

from ..research.recipes import RecipeResult, ResidualMomentumConfig, residual_momentum
from .. import strict_json

_MEMBER = 'F-F_Research_Data_Factors.csv'
_DATASET = 'french_us_monthly_3factor'
_BASIS = 'MKT_RF_is_market_excess_RF_is_same_month_return_not_annual_rate'


def _bytes(path: Path, sha: str, limit: int) -> bytes:
    if path.is_symlink():
        raise ValueError('french_artifact_symlink')
    with path.open('rb') as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit or hashlib.sha256(raw).hexdigest() != sha:
        raise ValueError('french_artifact_digest_or_size')
    return raw


def _json(raw):
    return strict_json.loads(raw)


def _months(raw):
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        if archive.namelist() != [_MEMBER] or archive.getinfo(_MEMBER).file_size > 10_000_000:
            raise ValueError('french_exact_bounded_member_required')
        reader = csv.reader(io.StringIO(archive.read(_MEMBER).decode('utf-8-sig')))
        for row in reader:
            if [cell.strip() for cell in row] == ['', 'Mkt-RF', 'SMB', 'HML', 'RF']:
                break
        else:
            raise ValueError('french_monthly_header_missing')
        result, previous = [], ''
        for row in reader:
            if not row or not row[0].strip():
                continue
            month = row[0].strip()
            if not re.fullmatch(r'[0-9]{6}', month):
                break
            if len(row) != 5 or month <= previous or len(result) >= 2400:
                raise ValueError('french_monthly_grid_invalid')
            previous = month
            year, number = int(month[:4]), int(month[4:])
            first = date(year, number, 1)
            last = date(year, number, calendar.monthrange(year, number)[1])
            values = [float(cell.strip()) for cell in row[1:]]
            if not all(math.isfinite(value) for value in values):
                raise ValueError('french_nonfinite_factor')
            missing = any(value in {-99.99, -999.0} for value in values)
            result.append({'month': month, 'periodStart': first.isoformat(), 'periodEnd': last.isoformat(),
                'values': {name: None if value in {-99.99, -999.0} else value / 100
                           for name, value in zip(('MKT_RF', 'SMB', 'HML', 'RF'), values, strict=True)},
                'missingReason': 'provider_missing_code' if missing else None})
        if not result:
            raise ValueError('french_monthly_rows_missing')
        return result


def load_french_factors(path: str | Path, *, sha256: str) -> dict:
    """Verify the projected monthly rows independently against original ZIP bytes."""
    path = Path(path)
    manifest = _json(_bytes(path, sha256, 2_000_000))
    required = {'contractVersion': 'french-monthly-inputs-v1', 'datasetId': _DATASET,
        'region': 'America', 'frequency': 'monthly', 'sourceUnit': 'percent',
        'normalizedUnit': 'decimal', 'parserVersion': 1, 'zipMember': _MEMBER, 'factorBasis': _BASIS}
    if any(manifest.get(key) != value for key, value in required.items()):
        raise ValueError('french_contract_invalid')
    if (any(manifest.get(key) is not False for key in
            ('historicalReceiptVerified', 'qualificationIssued', 'executionAuthorized'))
            or manifest.get('sourcePublishedAt') is not None
            or set(manifest['files']) != {'source-receipt.json', 'raw.zip'}):
        raise ValueError('french_evidence_contract_invalid')
    received, cutoff = strict_json.clock(manifest['receivedAt']), strict_json.clock(manifest['asOf'])
    if not received <= cutoff <= datetime.now(UTC):
        raise ValueError('french_receipt_cutoff_invalid')
    receipt = _json(_bytes(path.with_name('source-receipt.json'), manifest['files']['source-receipt.json'], 100_000))
    raw = _bytes(path.with_name('raw.zip'), manifest['files']['raw.zip'], 5_000_000)
    if (receipt.get('contractVersion') != 'research-source-receipt-v1'
            or receipt.get('source') != 'kenneth_french' or receipt.get('dataset') != _DATASET
            or receipt.get('format') != 'zip' or receipt.get('status') != 'received'
            or receipt.get('httpStatus') != 200 or receipt.get('rawFile') != 'raw.data'
            or receipt.get('historicalReceiptVerified') is not False
            or strict_json.clock(receipt['receivedAt']) != received
            or receipt['rawSha256'] != manifest['files']['raw.zip'] or receipt['rawBytes'] != len(raw)):
        raise ValueError('french_source_binding_invalid')
    start, end = (date.fromisoformat(manifest['scope'][key]) for key in ('start', 'end'))
    rows = [row for row in _months(raw) if start.isoformat() <= row['periodStart'] and row['periodEnd'] <= end.isoformat()]
    if (start > end or not rows or rows != manifest['rows']
            or any(date.fromisoformat(row['periodEnd']) >= received.date() for row in rows)):
        raise ValueError('french_monthly_projection_invalid')
    return manifest


def french_residual_momentum(
    returns: pl.DataFrame, *, path: str | Path, sha256: str,
    config: ResidualMomentumConfig, return_basis: str, return_currency: str,
    decision_at: datetime | None = None,
) -> RecipeResult:
    """Consume explicit YYYYMM simple-return periods and aware known_at clocks.

    Optional decision_at appends an empty decision row after the final month;
    skip_periods must exclude that row. Newly received vintages remain unavailable
    for earlier decisions. No timestamp-to-month guessing or return resampling.
    """
    manifest = load_french_factors(path, sha256=sha256)
    if return_currency != 'USD' or return_basis not in {'raw', 'excess'}:
        raise ValueError('french_USD_and_explicit_return_basis_required')
    if not isinstance(config, ResidualMomentumConfig):
        raise ValueError('french_residual_configuration_required')
    if 'month' not in returns.columns or 'known_at' not in returns.columns:
        raise ValueError('french_explicit_month_and_asset_clock_required')
    symbols = sorted(set(returns.columns) - {'month', 'known_at'})
    months = returns['month'].to_list()
    if (not symbols or not months or len(months) > 2400
            or any(not isinstance(month, str) or not re.fullmatch(r'[0-9]{6}', month) for month in months)
            or months != sorted(set(months))):
        raise ValueError('french_unique_ordered_months_required')
    factor_rows = {row['month']: row for row in manifest['rows']}
    asset_rows, aligned = [], []
    previous_end = None
    received, cutoff = strict_json.clock(manifest['receivedAt']), strict_json.clock(manifest['asOf'])
    for row in returns.iter_rows(named=True):
        month = row['month']
        if month not in factor_rows:
            raise ValueError('french_month_outside_frozen_input')
        factor = factor_rows[month]
        start = date.fromisoformat(factor['periodStart'])
        end = date.fromisoformat(factor['periodEnd']) + timedelta(days=1)
        if previous_end is not None and start != previous_end:
            raise ValueError('french_consecutive_months_required')
        previous_end = end
        close = datetime.combine(end, datetime.min.time(), UTC)
        known = row['known_at']
        if not isinstance(known, datetime) or known.utcoffset() is None or not close <= known <= cutoff:
            raise ValueError('french_asset_availability_invalid')
        asset_rows.append({'date': close, 'known_at': known, **{symbol: row[symbol] for symbol in symbols}})
        values = {key: value for key, value in factor['values'].items() if return_basis == 'raw' or key != 'RF'}
        aligned.append({'date': close, 'known_at': received, **values})
    if decision_at is not None:
        if (decision_at.utcoffset() is None or not asset_rows[-1]['date'] < decision_at <= cutoff
                or config.skip_periods < 1):
            raise ValueError('french_decision_clock_or_skip_invalid')
        asset_rows.append({'date': decision_at, 'known_at': decision_at, **dict.fromkeys(symbols)})
        aligned.append({'date': decision_at, 'known_at': max(received, decision_at),
                        **dict.fromkeys(key for key in aligned[-1] if key not in {'date', 'known_at'})})
    result = residual_momentum(pl.DataFrame(asset_rows, infer_schema_length=None),
        pl.DataFrame(aligned, infer_schema_length=None), config=config, return_basis=return_basis)
    return RecipeResult(result.values, result.diagnostics, {**result.summary,
        'factorInputSha256': sha256, 'factorReceivedAt': manifest['receivedAt'],
        'factorAsOf': manifest['asOf'], 'frequency': 'monthly', 'returnCurrency': 'USD',
        'periodConvention': 'explicit_YYYYMM_close_at_next_month_00_UTC',
        'historicalReceiptVerified': False, 'sourcePublishedAt': None,
        'decisionAt': decision_at.isoformat() if decision_at else None})
