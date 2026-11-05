"""Select received frozen style snapshots per period without backfilling current exposures."""
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path

import polars as pl

from .equity_risk import load_equity_risk_inputs
from ..reporting.factor_risk import standardise_style_exposures

_DEFINITIONS = {'size': {'log_market_cap': 1.0}, 'value': {'book_to_market': 1.0}}


def received_style_history(bindings: list[dict], periods: list[dict], *,
                           model_at: datetime, target_instruments: list[int], method: str) -> dict:
    """Return period exposures and evidence selection from existing verified risk bundles.

    Each period explicitly lists its universe. No future/stale snapshot or subset
    standardisation is substituted. This does not certify historical security,
    currency, share-class or adjustment evidence absent from the source bundles.
    """
    if (not isinstance(model_at, datetime) or model_at.utcoffset() is None
            or model_at > datetime.now(UTC)
            or method not in {'OLS', 'WLS_sqrt_cap'} or not isinstance(bindings, list)
            or not 1 <= len(bindings) <= 1261 or not isinstance(periods, list) or not 1 <= len(periods) <= 1260):
        raise ValueError('dynamic_style_history_scope_invalid')
    def universe(ids):
        if (not isinstance(ids, list) or not 10 <= len(ids) <= 100 or len(set(ids)) != len(ids)
                or any(type(value) is not int or value <= 0 for value in ids)):
            raise ValueError('dynamic_style_explicit_bounded_universe_required')
        return sorted(str(value) for value in ids)
    target_symbols = universe(target_instruments)
    selected_periods, previous = [], None
    for period in periods:
        if set(period) != {'periodStart', 'instruments'}:
            raise ValueError('dynamic_style_period_binding_invalid')
        stamp = datetime.fromisoformat(period['periodStart'])
        if stamp.utcoffset() is None or stamp > model_at or previous is not None and stamp <= previous:
            raise ValueError('dynamic_style_ordered_nonfuture_periods_required')
        previous = stamp
        selected_periods.append((stamp, universe(period['instruments'])))
    snapshots, seen = [], set()
    total_bytes = 0
    for binding in bindings:
        if set(binding) != {'path', 'sha256'} or binding['sha256'] in seen:
            raise ValueError('dynamic_style_unique_snapshot_bindings_required')
        seen.add(binding['sha256'])
        path = Path(binding['path'])
        if any(part.is_symlink() for part in (path, *path.parents)) or path.stat().st_size > 2_000_000:
            raise ValueError('dynamic_style_snapshot_path_or_size_invalid')
        manifest_raw = path.read_bytes()
        if hashlib.sha256(manifest_raw).hexdigest() != binding['sha256']:
            raise ValueError('dynamic_style_snapshot_digest_mismatch')
        manifest = json.loads(manifest_raw)
        for name in manifest['files']:
            if Path(name).name != name or name in {'.', '..'}:
                raise ValueError('dynamic_style_snapshot_component_path_invalid')
            component = path.with_name(name)
            if component.is_symlink():
                raise ValueError('dynamic_style_snapshot_component_symlink')
            total_bytes += component.stat().st_size
            if total_bytes > 256_000_000:
                raise ValueError('dynamic_style_history_total_size_limit')
        data = load_equity_risk_inputs(path, sha256=binding['sha256'])
        stamp = datetime.fromisoformat(data.manifest['asOf'])
        if stamp > model_at:
            raise ValueError('dynamic_style_snapshot_after_model_cutoff')
        # Existing verification replays original profile and SEC facts, including clocks.
        exposures = standardise_style_exposures(data.characteristics, definitions=_DEFINITIONS)
        snapshots.append({'asOf': stamp, 'expiresAt': data.style_expires_at, 'sha256': binding['sha256'],
                          'exposures': exposures, 'characteristics': data.characteristics,
                          'admission': data.execution_admission})
    evidence, rows = [], []
    def select(stamp, symbols, *, target=False):
        eligible = [snapshot for snapshot in snapshots if snapshot['asOf'] <= stamp]
        reason, chosen = None, None
        if not eligible:
            reason = 'no_snapshot_received_before_period'
        else:
            latest = max(snapshot['asOf'] for snapshot in eligible)
            candidates = [snapshot for snapshot in eligible if snapshot['asOf'] == latest]
            if len(candidates) != 1:
                reason = 'ambiguous_latest_snapshot'
            else:
                chosen = candidates[0]
                if stamp > chosen['expiresAt']:
                    reason = 'latest_snapshot_expired'
                elif sorted(chosen['exposures']['symbol'].to_list()) != symbols:
                    reason = 'snapshot_standardization_universe_mismatch'
        evidence.append({'timestamp': stamp.isoformat(), 'role': 'target' if target else 'period',
            'inputSha256': chosen['sha256'] if chosen else None, 'reason': reason,
            'historicalSemanticsCertified': False,
            'historicalSemanticsVersion': (chosen['admission'].get('historicalInputs') or {}).get('version') if chosen else None})
        if reason or chosen is None:
            return [{'timestamp' if target else 'period_start': stamp, 'symbol': symbol, 'known_at': stamp,
                     'size': None, 'value': None,
                     **({'market_cap': None} if not target and method == 'WLS_sqrt_cap' else {})} for symbol in symbols]
        caps = dict(zip(chosen['characteristics']['symbol'], chosen['characteristics']['market_cap'], strict=True))
        output = []
        for row in chosen['exposures'].iter_rows(named=True):
            output.append({'timestamp' if target else 'period_start': stamp, 'symbol': row['symbol'],
                'known_at': max(row['known_at'], chosen['asOf']), 'size': row['size'], 'value': row['value'],
                **({'market_cap': caps[row['symbol']]} if not target and method == 'WLS_sqrt_cap' else {})})
        return output
    for stamp, symbols in selected_periods:
        rows.extend(select(stamp, symbols))
    targets = select(model_at, target_symbols, target=True)
    metadata = {'contractVersion': 'received-style-history-v1', 'analysisMode': 'historical_dynamic',
                'modelAt': model_at.isoformat(), 'definitions': _DEFINITIONS,
                'standardization': 'winsor_0.025_cap_center_equal_weight_sample_scale',
                'selections': evidence, 'qualificationIssued': False, 'executionAuthorized': False,
                'historicalInputsCertified': False}
    raw = json.dumps(metadata, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return {'exposures': pl.DataFrame(rows, infer_schema_length=None),
            'targetExposures': pl.DataFrame(targets, infer_schema_length=None),
            'evidence': metadata, 'selectionSha256': hashlib.sha256(raw).hexdigest()}


def _frozen_bytes(path: Path, sha256: str, *, limit: int) -> bytes:
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError('dynamic_style_frozen_symlink')
    with path.open('rb') as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit or hashlib.sha256(raw).hexdigest() != sha256:
        raise ValueError('dynamic_style_frozen_digest_or_size')
    return raw


def freeze_style_history(output: str | Path, bindings: list[dict], periods: list[dict], *,
                         model_at: datetime, target_instruments: list[int], method: str) -> dict:
    """Copy original evidence to a new self-contained bundle; publish its manifest last."""
    from .equity_risk import _load
    import re
    import tempfile

    output = Path(output)
    # Verify semantic source bindings before creating user-visible output.
    selection = received_style_history(bindings, periods, model_at=model_at,
                                       target_instruments=target_instruments, method=method)
    with tempfile.TemporaryDirectory(prefix='dynamic-style-freeze-') as temporary:
        stage = Path(temporary).resolve()
        files, frozen_bindings, total = {}, [], 0
        for binding in bindings:
            source = Path(binding['path'])
            raw = _frozen_bytes(source, binding['sha256'], limit=2_000_000)
            snapshot = _load(raw)
            folder = 'snapshots/' + binding['sha256']
            if not re.fullmatch(r'[0-9a-f]{64}', binding['sha256']):
                raise ValueError('dynamic_style_snapshot_digest_invalid')
            members = {'manifest.json': binding['sha256'], **snapshot['files']}
            for name, digest in members.items():
                if not re.fullmatch(r'[A-Za-z0-9_.-]+', name) or name in {'.', '..'}:
                    raise ValueError('dynamic_style_snapshot_component_path_invalid')
                content = raw if name == 'manifest.json' else _frozen_bytes(source.with_name(name), digest,
                                                                                 limit=256_000_000 - total)
                total += len(content)
                if total > 256_000_000:
                    raise ValueError('dynamic_style_history_total_size_limit')
                relative = folder + '/' + name
                target = stage / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
                files[relative] = digest
            frozen_bindings.append({'path': folder + '/manifest.json', 'sha256': binding['sha256']})
        manifest = {'contractVersion': 'frozen-style-history-v1', 'modelAt': model_at.isoformat(),
                    'periods': periods, 'targetInstruments': target_instruments, 'method': method,
                    'snapshots': frozen_bindings, 'files': files,
                    'selectionSha256': selection['selectionSha256'],
                    'qualificationIssued': False, 'executionAuthorized': False}
        raw = json.dumps(manifest, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
        digest = hashlib.sha256(raw).hexdigest()
        (stage / 'manifest.json').write_bytes(raw)
        # Validate copied bytes too; no dependence on source directory survives publication.
        load_style_history(stage / 'manifest.json', sha256=digest)
        output.mkdir(parents=True, exist_ok=False, mode=0o700)
        import shutil
        for relative in files:
            target = output / relative
            target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            with target.open('xb') as stream, (stage / relative).open('rb') as source:
                shutil.copyfileobj(source, stream)
        with (output / 'manifest.json').open('xb') as stream:
            stream.write(raw)
    return {'path': str((output / 'manifest.json').resolve()), 'sha256': digest,
            'selectionSha256': selection['selectionSha256']}


def load_style_history(path: str | Path, *, sha256: str) -> dict:
    """Rebuild period selection solely from frozen original receipts and bound settings."""
    from .equity_risk import _load
    import re

    path = Path(path)
    manifest = _load(_frozen_bytes(path, sha256, limit=2_000_000))
    if (set(manifest) != {'contractVersion', 'modelAt', 'periods', 'targetInstruments', 'method',
                         'snapshots', 'files', 'selectionSha256', 'qualificationIssued', 'executionAuthorized'}
            or manifest['contractVersion'] != 'frozen-style-history-v1'
            or manifest['qualificationIssued'] is not False or manifest['executionAuthorized'] is not False
            or not isinstance(manifest['snapshots'], list) or not 1 <= len(manifest['snapshots']) <= 1261):
        raise ValueError('dynamic_style_frozen_contract_invalid')
    bindings, expected, total = [], {}, 0
    for binding in manifest['snapshots']:
        if (set(binding) != {'path', 'sha256'} or not isinstance(binding['sha256'], str)
                or not re.fullmatch(r'[0-9a-f]{64}', binding['sha256'])
                or binding['path'] != 'snapshots/' + binding['sha256'] + '/manifest.json'
                or binding['path'] in expected):
            raise ValueError('dynamic_style_frozen_snapshot_reference_invalid')
        target = path.parent / binding['path']
        raw = _frozen_bytes(target, binding['sha256'], limit=2_000_000)
        total += len(raw)
        snapshot = _load(raw)
        expected[binding['path']] = binding['sha256']
        for name, digest in snapshot['files'].items():
            if not re.fullmatch(r'[A-Za-z0-9_.-]+', name) or name in {'.', '..', 'manifest.json'}:
                raise ValueError('dynamic_style_frozen_component_path_invalid')
            content = _frozen_bytes(target.with_name(name), digest, limit=max(0, 256_000_000 - total))
            total += len(content)
            if total > 256_000_000:
                raise ValueError('dynamic_style_history_total_size_limit')
            expected['snapshots/' + binding['sha256'] + '/' + name] = digest
        bindings.append({'path': str(target), 'sha256': binding['sha256']})
    if expected != manifest['files']:
        raise ValueError('dynamic_style_frozen_component_grid_mismatch')
    selection = received_style_history(bindings, manifest['periods'],
        model_at=datetime.fromisoformat(manifest['modelAt']),
        target_instruments=manifest['targetInstruments'], method=manifest['method'])
    if selection['selectionSha256'] != manifest['selectionSha256']:
        raise ValueError('dynamic_style_frozen_selection_mismatch')
    return {**selection, 'inputSha256': sha256, 'manifest': manifest}


def estimate_frozen_dynamic_style(path: str | Path, *, sha256: str,
                                  returns_path: str | Path, returns_sha256: str,
                                  window: int, minimum_observations: int, model_id: str) -> dict:
    """Bind frozen style selections to verified received return intervals and estimate."""
    from ..reporting.dynamic_style_risk import estimate_dynamic_style_risk

    if (type(window) is not int or type(minimum_observations) is not int
            or not 2 <= minimum_observations <= window <= 1260):
        raise ValueError('dynamic_style_window_configuration_invalid')
    history = load_style_history(path, sha256=sha256)
    source = load_equity_risk_inputs(returns_path, sha256=returns_sha256)
    from ..reporting.risk_model import _model_identity
    _model_identity(model_id, source.manifest['currency'], returns_sha256)
    manifest = history['manifest']
    model_at = datetime.fromisoformat(manifest['modelAt'])
    if datetime.fromisoformat(source.manifest['asOf']) > model_at:
        raise ValueError('dynamic_return_snapshot_after_model_cutoff')
    panel = source.period_returns
    if panel is None:
        raise ValueError('dynamic_explicit_received_return_intervals_required')
    request = {'styleHistorySha256': sha256, 'returnsInputSha256': returns_sha256,
               'window': window, 'minimumObservations': minimum_observations,
               'modelId': model_id, 'method': manifest['method']}
    input_digest = hashlib.sha256(json.dumps(request, sort_keys=True, separators=(',', ':'),
                                           allow_nan=False).encode()).hexdigest()
    lookup = {(row['period_start'], row['symbol']): row for row in panel.iter_rows(named=True)}
    rows, gaps = [], []
    for period in manifest['periods']:
        start = datetime.fromisoformat(period['periodStart'])
        for instrument in period['instruments']:
            row = lookup.get((start, str(instrument)))
            if row is None:
                gaps.append({'periodStart': period['periodStart'], 'instrumentId': instrument,
                             'reason': 'missing_received_return_interval'})
            else:
                rows.append(row)
    identity = {**request, 'inputSha256': input_digest, 'selectionSha256': history['selectionSha256'],
                'analysisMode': 'historical_dynamic', 'historicalInputsCertified': False,
                'qualificationIssued': False, 'executionAuthorized': False}
    if gaps:
        return {**identity, 'status': 'unavailable', 'reason': 'incomplete_received_return_grid',
                'gaps': gaps, 'model': None, 'styleEvidence': history['evidence']}
    result = estimate_dynamic_style_risk(pl.DataFrame(rows), history['exposures'],
        history['targetExposures'], factors=['size', 'value'], method=manifest['method'],
        window=window, minimum_observations=minimum_observations,
        fit_end=max(row['timestamp'] for row in rows), model_known_at=model_at,
        currency=source.manifest['currency'], input_sha256=input_digest, model_id=model_id)
    exposure_history = history['exposures'].with_columns(
        pl.col('period_start', 'known_at').dt.to_string('%+')).to_dicts()
    return {**result, **identity, 'styleEvidence': history['evidence'], 'exposureHistory': exposure_history,
            'returnEvidence': {'basis': 'received_total_return_price_ratios',
                               'periodStartBasis': 'previous_received_session_close',
                               'historicalSemanticsCertified': False}}
