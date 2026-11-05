"""Named recipe research entry points using digest-bound inputs and existing algorithms."""
from __future__ import annotations

import hashlib
import base64
import json
import platform
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

import polars as pl

from ..adapters.french_factors import _bytes, _json, french_residual_momentum
from ..event.manifest import engine_code_digest
from ..research.recipes import ResidualMomentumConfig


def _encode(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False, indent=2).encode()


def french_residual_command(args):
    """Compute recipe scores and optional portfolio/event results using existing modules."""
    if (args.allocation is None) != (args.allocation_sha256 is None):
        raise ValueError('recipe_allocation_path_and_digest_required_together')
    if (args.execution is None) != (args.execution_sha256 is None):
        raise ValueError('recipe_execution_path_and_digest_required_together')
    if args.execution is not None and args.allocation is None:
        raise ValueError('recipe_execution_requires_allocation')
    engine_code_digest.cache_clear()
    engine_sha = engine_code_digest()
    method = {'engineCodeSha256': engine_sha, 'python': platform.python_version(),
              'numpy': version('numpy'), 'polars': version('polars')}
    if args.expected_method_json is not None and _json(args.expected_method_json) != method:
        raise ValueError('recipe_method_version_mismatch')
    raw = _bytes(args.returns, args.returns_sha256, 32_000_000)
    document = _json(raw)
    if (set(document) != {'contractVersion', 'returnBasis', 'currency', 'rows'}
            or document['contractVersion'] != 'monthly-asset-returns-v1'
            or not isinstance(document['rows'], list) or not 1 <= len(document['rows']) <= 2400):
        raise ValueError('recipe_monthly_returns_contract_invalid')
    rows, symbols = [], None
    for row in document['rows']:
        if not isinstance(row, dict) or set(row) != {'month', 'knownAt', 'values'}:
            raise ValueError('recipe_monthly_return_row_invalid')
        values = row['values']
        if not isinstance(values, dict) or not 1 <= len(values) <= 512:
            raise ValueError('recipe_asset_count_outside_1_512')
        if symbols is None:
            symbols = set(values)
        if set(values) != symbols or any(key in {'month', 'known_at', 'date'} for key in values):
            raise ValueError('recipe_return_asset_grid_changed')
        if any(value is not None and type(value) not in {int, float} for value in values.values()):
            raise ValueError('recipe_numeric_returns_required')
        rows.append({'month': row['month'], 'known_at': datetime.fromisoformat(row['knownAt']), **values})
    config = ResidualMomentumConfig(args.training_periods, args.scoring_periods, args.skip_periods)
    # Freeze exact bytes before calculating so the persisted inputs are those consumed.
    manifest_raw = _bytes(args.factors, args.factors_sha256, 2_000_000)
    manifest = _json(manifest_raw)
    if set(manifest['files']) != {'raw.zip', 'source-receipt.json'}:
        raise ValueError('recipe_french_bundle_files_invalid')
    files = {'returns.json': raw, 'factors/manifest.json': manifest_raw}
    for name, limit in [('raw.zip', 5_000_000), ('source-receipt.json', 100_000)]:
        files['factors/' + name] = _bytes(args.factors.with_name(name), manifest['files'][name], limit)
    import tempfile
    with tempfile.TemporaryDirectory(prefix='cortex-french-recipe-') as temporary:
        frozen = Path(temporary)
        for name, content in files.items():
            target = frozen / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        result = french_residual_momentum(pl.DataFrame(rows, infer_schema_length=None),
            path=frozen / 'factors/manifest.json', sha256=args.factors_sha256,
            config=config, return_basis=document['returnBasis'], return_currency=document['currency'],
            decision_at=datetime.fromisoformat(args.decision_at) if args.decision_at else None)
    portfolio = None
    if args.allocation is not None:
        from ..research.holding import HoldingConfig
        from ..research.recipe_portfolio import residual_score_portfolio
        allocation_raw = _bytes(args.allocation, args.allocation_sha256, 100_000)
        allocation = _json(allocation_raw)
        if set(allocation) != {'universeKnownAt', 'topN', 'longShort', 'maxGross', 'holding'}:
            raise ValueError('recipe_allocation_contract_invalid')
        controls = allocation['holding']
        allowed = {'enter_periods', 'exit_periods', 'min_active_periods', 'cooldown_periods',
                   'holding_periods', 'formation_interval'}
        if not isinstance(controls, dict) or set(controls) - allowed:
            raise ValueError('recipe_selection_holding_controls_invalid')
        portfolio = residual_score_portfolio(result.values,
            universe_known_at=datetime.fromisoformat(allocation['universeKnownAt']),
            top_n=allocation['topN'], long_short=allocation['longShort'], max_gross=allocation['maxGross'],
            holding=HoldingConfig(max_gross=allocation['maxGross'], **controls))
        files['allocation.json'] = allocation_raw
        files['portfolio.json'] = _encode(portfolio)
    execution = None
    if args.execution is not None:
        if portfolio is None:
            raise ValueError('recipe_execution_requires_allocation')
        from ..adapters.recipe_execution import execute_recipe_document
        execution_raw = _bytes(args.execution, args.execution_sha256, 64_000_000)
        execution_doc = _json(execution_raw)
        # Bind portfolio, factor and asset inputs as well as execution prices/costs.
        binding = {'execution': args.execution_sha256, 'portfolio': hashlib.sha256(files['portfolio.json']).hexdigest(),
                   'factors': args.factors_sha256, 'returns': args.returns_sha256}
        run = execute_recipe_document(execution_doc, portfolio,
            input_sha256=hashlib.sha256(_encode(binding)).hexdigest())
        execution = run.report(include_records=True)
        execution['recipeInputs'] = binding
        last_open = max(datetime.fromisoformat(row['openAt']) for row in execution_doc['bars'])
        execution['decisionCoverage'] = {
            'lastExecutionOpen': last_open.isoformat(),
            'decisionsWithoutLaterOpen': sum(datetime.fromisoformat(row['date']) >= last_open
                                             for row in portfolio['values']),
            'meaning': 'targets_without_a_later_open_are_not_realized_trades'}
        execution['inputEvidence'] = 'caller_supplied_raw_equity_bars_and_actions_not_provider_verified'
        execution['qualificationIssued'] = False
        execution['executionAuthorized'] = False
        files['execution-input.json'] = execution_raw
        files['execution-report.json'] = _encode(execution)
        for name, frame in {'equity': run.performance.equity_curve(),
                            'orders': run.records.orders(), 'fills': run.records.fills()}.items():
            files[name + '.csv'] = frame.write_csv().encode()
    engine_code_digest.cache_clear()
    if engine_code_digest() != engine_sha:
        raise ValueError('recipe_engine_changed_during_calculation')
    report = result.to_dict()
    report['summary'].update({'recipeId': 'french_monthly_out_of_fit_residual_momentum',
        'assetReturnsSha256': args.returns_sha256, 'outputScope': 'scores_and_diagnostics_only',
        'assetReturnEvidence': 'caller_supplied_monthly_returns_not_provider_verified'})
    output_scope = 'scores_and_target_weights' if portfolio is not None else 'scores_and_diagnostics_only'
    if execution is not None:
        output_scope = 'scores_targets_and_simulated_execution'
        report['execution'] = execution
    report['summary']['outputScope'] = output_scope
    if portfolio is not None:
        report['portfolio'] = portfolio
    files['report.json'] = _encode(report)
    request = {'contractVersion': 'recipe-research-v1',
        'recipeId': 'french_monthly_out_of_fit_residual_momentum',
        'returns': {'path': 'returns.json', 'sha256': args.returns_sha256},
        'factors': {'path': 'factors/manifest.json', 'sha256': args.factors_sha256},
        'allocation': {'path': 'allocation.json', 'sha256': args.allocation_sha256} if portfolio is not None else None,
        'execution': {'path': 'execution-input.json', 'sha256': args.execution_sha256} if execution is not None else None,
        'trainingPeriods': args.training_periods, 'scoringPeriods': args.scoring_periods,
        'skipPeriods': args.skip_periods, 'decisionAt': args.decision_at,
        'method': method}
    input_names = ['returns.json', 'factors/manifest.json', 'factors/raw.zip', 'factors/source-receipt.json']
    if portfolio is not None:
        input_names.append('allocation.json')
    if execution is not None:
        input_names.append('execution-input.json')
    package = {'contractVersion': 'recipe-input-package-v1', 'request': request,
               'files': {name: base64.b64encode(files[name]).decode('ascii') for name in input_names}}
    package_raw = _encode(package)
    if len(package_raw) > 96_000_000:
        raise ValueError('recipe_input_package_size_limit')
    files['input-package.json'] = package_raw
    files['request.json'] = _encode(request)
    completion = {'contractVersion': 'recipe-research-completion-v1',
        'files': {name: hashlib.sha256(content).hexdigest() for name, content in files.items()},
        'qualificationIssued': False, 'executionAuthorized': False}
    args.output.mkdir(parents=True, exist_ok=False)
    for name, content in files.items():
        target = args.output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    completion_bytes = _encode(completion)
    (args.output / 'completed.json').write_bytes(completion_bytes)
    return {'inputPackageSha256': hashlib.sha256(package_raw).hexdigest(), 'path': str(args.output.resolve()), 'completionSha256': hashlib.sha256(completion_bytes).hexdigest(),
            'recipeId': request['recipeId'], 'outputScope': output_scope,
            'qualificationIssued': False, 'executionAuthorized': False}


def add_recipe_commands(subparsers):
    parser = subparsers.add_parser('recipe-french-residual',
        help='Compute monthly residual scores from frozen French factors and explicit USD return periods.')
    parser.add_argument('--expected-method-json', help='Require the frozen engine and runtime identities before computation.')
    parser.add_argument('--returns', required=True, type=Path,
        help='monthly-asset-returns-v1 JSON: returnBasis raw/excess, currency USD, rows month/knownAt/values.')
    parser.add_argument('--returns-sha256', required=True)
    parser.add_argument('--factors', required=True, type=Path, help='Frozen french-monthly-inputs-v1 manifest.')
    parser.add_argument('--factors-sha256', required=True)
    parser.add_argument('--training-periods', required=True, type=int)
    parser.add_argument('--scoring-periods', required=True, type=int)
    parser.add_argument('--skip-periods', required=True, type=int)
    parser.add_argument('--decision-at', help='Optional aware decision timestamp; requires skip-periods >= 1.')
    parser.add_argument('--allocation', type=Path, help='Optional pinned-universe rank/holding JSON.')
    parser.add_argument('--allocation-sha256')
    parser.add_argument('--execution', type=Path, help='Optional recipe-equity-execution-v1 raw bars/actions/costs JSON.')
    parser.add_argument('--execution-sha256')
    parser.add_argument('--output', required=True, type=Path, help='New result directory; existing paths are refused.')
    parser.set_defaults(func=french_residual_command)
