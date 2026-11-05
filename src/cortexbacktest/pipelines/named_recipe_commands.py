"""Bounded consumers for the retained frozen-pair and raw-contract curve recipes."""
from datetime import datetime, timedelta
import base64
import hashlib
import platform
from importlib.metadata import version
from pathlib import Path
import polars as pl
from ..adapters.french_factors import _bytes, _json
from ..event.manifest import engine_code_digest
from ..research.recipes import RecipeResult
from .recipe_commands import _encode

RECIPES = {'frozen_equity_pairs', 'raw_contract_term_structure'}


def _prices(rows):
    if not isinstance(rows, list) or not 1 <= len(rows) <= 5000:
        raise ValueError('pair_price_row_limit')
    output, symbols, previous = [], None, None
    for row in rows:
        if set(row) != {'date', 'knownAt', 'values'} or not isinstance(row['values'], dict):
            raise ValueError('pair_dated_price_row_required')
        stamp, known = datetime.fromisoformat(row['date']), datetime.fromisoformat(row['knownAt'])
        if stamp.utcoffset() is None or known.utcoffset() is None or known < stamp or (previous and stamp <= previous):
            raise ValueError('pair_price_clock_invalid')
        current = set(row['values'])
        if (not 2 <= len(current) <= 100 or current & {'date', 'known_at'}
                or (symbols is not None and current != symbols)
                or any(v is not None and type(v) not in {int, float} for v in row['values'].values())):
            raise ValueError('pair_price_asset_grid_invalid')
        symbols, previous = current, stamp
        output.append({'date': stamp, 'known_at': known, **row['values']})
    return pl.DataFrame(output, infer_schema_length=None)


def calculate(recipe_id, doc):
    if recipe_id == 'raw_contract_term_structure':
        from ..research.term_structure import term_structure
        if set(doc) != {'contractVersion', 'rows', 'maxQuoteAgeSeconds', 'minDaysToLastTrade', 'annualized'} or doc['contractVersion'] != 'raw-contract-recipe-v1':
            raise ValueError('raw_contract_recipe_input_invalid')
        if not isinstance(doc['rows'], list) or not 1 <= len(doc['rows']) <= 100000:
            raise ValueError('raw_contract_recipe_row_limit')
        if type(doc['maxQuoteAgeSeconds']) is not int or not 0 <= doc['maxQuoteAgeSeconds'] <= 86400:
            raise ValueError('raw_contract_recipe_quote_age_invalid')
        rows = [{**row, **{key: datetime.fromisoformat(row[key]) for key in ('date','expiry','last_trade_at','quote_at','known_at')}} for row in doc['rows']]
        return term_structure(pl.DataFrame(rows, infer_schema_length=None),
            max_quote_age=timedelta(seconds=doc['maxQuoteAgeSeconds']), min_days_to_last_trade=doc['minDaysToLastTrade'], annualized=doc['annualized']).to_dict(), None
    from ..research.pairs import PairConfig, PairRecipe, fit_pairs
    if set(doc) != {'contractVersion', 'currency', 'formationKnownAt', 'formation', 'evaluation', 'config'} or doc['contractVersion'] != 'frozen-equity-pair-recipe-v1' or doc['currency'] != 'USD':
        raise ValueError('frozen_equity_pair_input_invalid')
    config = PairConfig(**doc['config'])
    formation, evaluation = _prices(doc['formation']), _prices(doc['evaluation'])
    if set(formation.columns) != set(evaluation.columns) or evaluation['date'][0] <= formation['date'][-1]:
        raise ValueError('pair_formation_evaluation_universe_mismatch')
    fit = fit_pairs(formation, config=config, known_at=datetime.fromisoformat(doc['formationKnownAt']), currency=doc['currency'])
    diagnostics = fit.candidates.to_dicts()
    models = [model.to_document() for model in fit.models]
    if not fit.models:
        return {'summary': {'status': 'unavailable', 'reason': 'no_eligible_frozen_pairs', 'formation': fit.summary},
                'values': [], 'diagnostics': diagnostics, 'models': [], 'qualificationIssued': False, 'executionAuthorized': False}, None
    recipe = PairRecipe(fit.models, config=config)
    values, last = [], None
    for row in evaluation.iter_rows(named=True):
        # Received bar values become usable at knownAt, never at their earlier session label.
        decision = row['known_at']
        if last is not None and decision <= last:
            raise ValueError('pair_decision_clocks_must_increase')
        last = decision
        weights = recipe.update(decision, {symbol: row[symbol] for symbol in recipe.symbols})
        # Preserve the entire execution universe, including explicit zero weights for unselected assets.
        values.append({'date': decision, **{s: weights.get(s, 0.) for s in evaluation.columns if s not in {'date','known_at'}}})
    portfolio = RecipeResult(pl.DataFrame(values), pl.DataFrame(), {
        'output': 'target_weights_not_fills_or_realized_returns', 'currency': 'USD',
        'formation': fit.summary, 'maxGross': config.max_gross, 'decisionClock': 'received_bar_known_at',
        'universeEvidence': 'caller_pinned_not_historical_membership_certification'}).to_dict()
    return {**portfolio, 'models': models, 'diagnostics': diagnostics, 'checkpoint': recipe.to_document()}, portfolio


def named_recipe_command(args):
    if args.recipe_id not in RECIPES:
        raise ValueError('unsupported_named_recipe')
    if (args.execution is None) != (args.execution_sha256 is None):
        raise ValueError('recipe_execution_path_and_digest_required_together')
    if args.execution is not None and args.recipe_id != 'frozen_equity_pairs':
        raise ValueError('raw_contract_curve_scores_are_not_equity_execution_targets')
    engine_code_digest.cache_clear()
    method = {'engineCodeSha256': engine_code_digest(), 'python': platform.python_version(), 'numpy': version('numpy'), 'polars': version('polars')}
    if args.expected_method_json is not None and _json(args.expected_method_json) != method:
        raise ValueError('recipe_method_version_mismatch')
    raw = _bytes(args.input, args.input_sha256, 32_000_000)
    report, portfolio = calculate(args.recipe_id, _json(raw))
    inputs, outputs = {'recipe-input.json': raw}, {}
    if args.execution is not None:
        from ..adapters.recipe_execution import execute_recipe_document
        if portfolio is None:
            raise ValueError('recipe_execution_requires_available_pair_targets')
        execution_raw = _bytes(args.execution, args.execution_sha256, 64_000_000)
        inputs['execution-input.json'] = execution_raw
        binding = {'input': args.input_sha256, 'execution': args.execution_sha256,
                   'targets': hashlib.sha256(_encode(portfolio)).hexdigest()}
        execution_doc = _json(execution_raw)
        run = execute_recipe_document(execution_doc, portfolio, input_sha256=hashlib.sha256(_encode(binding)).hexdigest())
        execution = run.report(include_records=True)
        last_open = max(datetime.fromisoformat(row['openAt']) for row in execution_doc['bars'])
        execution.update({'recipeInputs': binding, 'qualificationIssued': False, 'executionAuthorized': False,
            'decisionCoverage': {'lastExecutionOpen': last_open.isoformat(), 'decisionsWithoutLaterOpen': sum(datetime.fromisoformat(row['date']) >= last_open for row in portfolio['values'])}})
        report['execution'] = execution
        outputs['execution-report.json'] = _encode(execution)
        for name, frame in {'equity': run.performance.equity_curve(), 'orders': run.records.orders(), 'fills': run.records.fills()}.items():
            outputs[name + '.csv'] = frame.write_csv().encode()
    if portfolio is not None:
        report['portfolio'] = portfolio
    scope = 'targets_and_simulated_execution' if args.execution is not None else 'target_weights' if portfolio is not None else 'curve_scores_and_diagnostics' if args.recipe_id == 'raw_contract_term_structure' else 'unavailable'
    report['summary'].update({'recipeId': args.recipe_id, 'recipeInputSha256': args.input_sha256,
                              'outputScope': scope, 'inputEvidence': 'caller_supplied_frozen_inputs_not_vendor_certified'})
    request = {'contractVersion': 'named-recipe-research-v1', 'recipeId': args.recipe_id,
        'input': {'path': 'recipe-input.json', 'sha256': args.input_sha256},
        'execution': {'path': 'execution-input.json', 'sha256': args.execution_sha256} if args.execution else None, 'method': method}
    package = {'contractVersion': 'recipe-input-package-v1', 'request': request,
               'files': {name: base64.b64encode(content).decode() for name, content in inputs.items()}}
    package_raw = _encode(package)
    if len(package_raw) > 96_000_000:
        raise ValueError('recipe_input_package_size_limit')
    engine_code_digest.cache_clear()
    if engine_code_digest() != method['engineCodeSha256']:
        raise ValueError('recipe_engine_changed_during_calculation')
    outputs.update(inputs)
    outputs.update({'report.json': _encode(report), 'request.json': _encode(request), 'input-package.json': package_raw})
    completion = {'contractVersion': 'recipe-research-completion-v1', 'files': {n: hashlib.sha256(v).hexdigest() for n,v in outputs.items()},
                  'qualificationIssued': False, 'executionAuthorized': False}
    args.output.mkdir(parents=True, exist_ok=False, mode=0o700)
    for name, content in outputs.items():
        (args.output / name).write_bytes(content)
    (args.output / 'completed.json').write_bytes(_encode(completion))
    return {'path': str(args.output), 'recipeId': args.recipe_id, 'inputPackageSha256': hashlib.sha256(package_raw).hexdigest(),
            'qualificationIssued': False, 'executionAuthorized': False}


def add_named_recipe_commands(subparsers):
    parser = subparsers.add_parser('recipe-named', help='Run frozen equity pairs or raw-contract term-structure research.')
    parser.add_argument('--recipe-id', choices=sorted(RECIPES), required=True)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--input-sha256', required=True)
    parser.add_argument('--execution', type=Path)
    parser.add_argument('--execution-sha256')
    parser.add_argument('--expected-method-json')
    parser.add_argument('--output', type=Path, required=True)
    parser.set_defaults(func=named_recipe_command)
