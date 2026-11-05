"""Existing CLI entry points for receipt-bound historical dynamic style research."""
from datetime import datetime
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import re
import shutil
import tempfile

from ..adapters.dynamic_style_history import (
    _frozen_bytes, freeze_style_history, load_style_history, estimate_frozen_dynamic_style,
)
from ..adapters.equity_risk import _load, load_equity_risk_inputs
from ..event.manifest import engine_code_digest


def _encode(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def freeze_command(args):
    spec = _load(_frozen_bytes(args.spec, args.spec_sha256, limit=4_000_000))
    if set(spec) != {'snapshots', 'periods', 'modelAt', 'targetInstruments', 'method'}:
        raise ValueError('dynamic_style_freeze_spec_invalid')
    return freeze_style_history(args.output, spec['snapshots'], spec['periods'],
        model_at=datetime.fromisoformat(spec['modelAt']), target_instruments=spec['targetInstruments'],
        method=spec['method'])


def estimate_command(args):
    engine_code_digest.cache_clear()
    before = engine_code_digest()
    history = load_style_history(args.history, sha256=args.history_sha256)
    source = load_equity_risk_inputs(args.returns, sha256=args.returns_sha256)
    with tempfile.TemporaryDirectory(prefix='dynamic-style-run-') as temporary:
        stage = Path(temporary).resolve()
        files, total = {}, 0
        for prefix, path, digest, manifest in [
            ('history', args.history, args.history_sha256, history['manifest']),
            ('returns', args.returns, args.returns_sha256, source.manifest),
        ]:
            for name, expected in {'manifest.json': digest, **manifest['files']}.items():
                valid = (re.fullmatch(r'snapshots/[0-9a-f]{64}/[A-Za-z0-9_.-]+', name)
                         if prefix == 'history' and name != 'manifest.json'
                         else re.fullmatch(r'[A-Za-z0-9_.-]+', name))
                if not valid or Path(name).name in {'.', '..'}:
                    raise ValueError('dynamic_style_run_component_path_invalid')
                origin = path if name == 'manifest.json' else path.parent / name
                raw = _frozen_bytes(origin, expected, limit=max(0, 512_000_000 - total))
                total += len(raw)
                if total > 512_000_000:
                    raise ValueError('dynamic_style_run_size_limit')
                relative = prefix + '/' + name
                target = stage / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(raw)
                files[relative] = expected
        result = estimate_frozen_dynamic_style(stage / 'history/manifest.json', sha256=args.history_sha256,
            returns_path=stage / 'returns/manifest.json', returns_sha256=args.returns_sha256,
            window=args.window, minimum_observations=args.minimum_observations, model_id=args.model_id)
        engine_code_digest.cache_clear()
        if engine_code_digest() != before:
            raise ValueError('dynamic_style_engine_changed_during_run')
        request = {'contractVersion': 'dynamic-style-run-v1',
            'history': {'path': 'history/manifest.json', 'sha256': args.history_sha256},
            'returns': {'path': 'returns/manifest.json', 'sha256': args.returns_sha256},
            'window': args.window, 'minimumObservations': args.minimum_observations, 'modelId': args.model_id,
            'method': {'engineCodeSha256': before, 'python': platform.python_version(),
                       'numpy': version('numpy'), 'polars': version('polars')}}
        for name, document in [('request.json', request), ('report.json', result)]:
            raw = _encode(document)
            (stage / name).write_bytes(raw)
            files[name] = hashlib.sha256(raw).hexdigest()
        completion = {'contractVersion': 'dynamic-style-completion-v1', 'files': files,
                      'inputSha256': result['inputSha256'], 'qualificationIssued': False, 'executionAuthorized': False}
        completed_raw = _encode(completion)
        args.output.mkdir(parents=True, exist_ok=False, mode=0o700)
        for name in files:
            target = args.output / name
            target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            with target.open('xb') as stream, (stage / name).open('rb') as original:
                shutil.copyfileobj(original, stream)
        with (args.output / 'completed.json').open('xb') as stream:
            stream.write(completed_raw)
    return {'path': str(args.output.resolve()), 'status': result['status'], 'reason': result.get('reason'),
            'inputSha256': result['inputSha256'], 'completionSha256': hashlib.sha256(completed_raw).hexdigest(),
            'qualificationIssued': False, 'executionAuthorized': False}


def add_dynamic_style_commands(subparsers):
    task = subparsers.add_parser('dynamic-style-task', help='Run an owned frozen dynamic style task.')
    task.add_argument('--task', required=True, type=Path)
    task.add_argument('--task-sha256', required=True)
    task.add_argument('--output', required=True, type=Path)
    task.set_defaults(func=task_command)
    freeze = subparsers.add_parser('dynamic-style-freeze', help='Freeze received per-period style evidence.')
    freeze.add_argument('--spec', required=True, type=Path)
    freeze.add_argument('--spec-sha256', required=True)
    freeze.add_argument('--output', required=True, type=Path)
    freeze.set_defaults(func=freeze_command)
    estimate = subparsers.add_parser('dynamic-style-risk', help='Estimate dynamic style risk from frozen received inputs.')
    for name in ('history', 'returns'):
        estimate.add_argument('--' + name, required=True, type=Path)
        estimate.add_argument('--' + name + '-sha256', required=True)
    estimate.add_argument('--window', required=True, type=int)
    estimate.add_argument('--minimum-observations', required=True, type=int)
    estimate.add_argument('--model-id', required=True)
    estimate.add_argument('--output', required=True, type=Path)
    estimate.set_defaults(func=estimate_command)


def task_command(args):
    """Run the owned queue's frozen task without consulting live source jobs."""
    from types import SimpleNamespace
    task = _load(_frozen_bytes(args.task, args.task_sha256, limit=4_000_000))
    if task.get('contractVersion') != 'owned-dynamic-style-task-v1':
        raise ValueError('dynamic_style_task_contract_invalid')
    spec = task['historySpec']
    history = freeze_style_history(args.output.parent / 'dynamic-history', spec['snapshots'], spec['periods'],
        model_at=datetime.fromisoformat(spec['modelAt']), target_instruments=spec['targetInstruments'],
        method=spec['method'])
    scope = task['scope']
    return estimate_command(SimpleNamespace(history=Path(history['path']), history_sha256=history['sha256'],
        returns=Path(task['returns']['path']), returns_sha256=task['returns']['sha256'],
        window=scope['window'], minimum_observations=scope['minimum_observations'],
        model_id=scope['model_id'], output=args.output))
