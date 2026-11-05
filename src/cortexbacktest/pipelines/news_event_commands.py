"""Received US-equity news study, independent of model training."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import tempfile
import shutil
from importlib.metadata import version
import platform
import polars as pl
from ..adapters.dynamic_style_history import _frozen_bytes
from ..adapters.equity_risk import _load, load_equity_risk_inputs
from ..adapters.news_events import load_news_replay_events
from ..research.event_study import event_study
from ..event.manifest import engine_code_digest


def news_equity_command(args):
    request_raw = _frozen_bytes(args.request, args.request_sha256, limit=1_000_000)
    request = _load(request_raw)
    if request.get('kind') != 'native_news_event_study':
        raise ValueError('news_equity_request_required')
    origin = args.request.parent
    prices = load_equity_risk_inputs(origin / 'prices/manifest.json', sha256=request['price_source']['input_sha256'])
    news_raw = _frozen_bytes(origin / 'news.json', request['news_sha256'], limit=10_000_000)
    events, metadata = load_news_replay_events(origin / 'news.json', sha256=request['news_sha256'])
    selected = request['event_ids']
    if (not isinstance(selected, list) or not 1 <= len(selected) <= 50 or len(set(selected)) != len(selected)
            or not set(selected).issubset(events['event_id'].to_list()) or not metadata['receivedRevisionsVerified']):
        raise ValueError('selected_received_news_events_required')
    cutoff = datetime.fromisoformat(prices.manifest['asOf'])
    if datetime.fromisoformat(metadata['asOf']) > cutoff:
        raise ValueError('news_cutoff_after_price_cutoff')
    events = events.filter(pl.col('event_id').is_in(selected)).with_columns(pl.col('symbol').str.replace('^INSTRUMENT:', ''))
    benchmark = str(request['benchmark_instrument_id'])
    if benchmark not in prices.returns['symbol'].unique().to_list() or benchmark in events['symbol'].to_list():
        raise ValueError('distinct_frozen_news_target_and_benchmark_required')
    if set(events['symbol'].to_list()) - set(prices.returns['symbol'].to_list()):
        raise ValueError('news_target_not_in_price_input')
    engine_code_digest.cache_clear()
    engine = engine_code_digest()
    returns = prices.returns.filter(pl.col('known_at') <= cutoff).pivot(on='symbol', index='timestamp', values='return').rename({'timestamp': 'date'}).sort('date')
    result = event_study(returns, returns.select('date', pl.col(benchmark).alias('market')), events,
        as_of=cutoff, event_window=tuple(request['event_window']), estimation_window=request['estimation_window'],
        estimation_gap=request['estimation_gap'], model=request['model']).to_dict()
    method = {'engineCodeSha256': engine, 'python': platform.python_version(), 'numpy': version('numpy'), 'polars': version('polars')}
    report = {**result, 'status': 'available' if result['events'] else 'unavailable',
        'reason': None if result['events'] else 'no_accepted_events', 'requestedEvents': len(selected),
        'newsSha256': request['news_sha256'], 'newsSource': metadata,
        'priceInputSha256': request['price_source']['input_sha256'], 'priceInputAsOf': prices.manifest['asOf'],
        'benchmarkInstrumentId': request['benchmark_instrument_id'], 'method': method,
        'eventEvidence': 'received_news_receipts', 'returnBasis': 'received_total_return_price_ratios',
        'historicalInputsCertified': False}
    with tempfile.TemporaryDirectory(prefix='news-equity-study-') as temporary:
        stage = Path(temporary)
        (stage / 'prices').mkdir()
        files = {}
        for name, digest in {'manifest.json': request['price_source']['input_sha256'], **prices.manifest['files']}.items():
            if Path(name).name != name or name in {'.', '..'}:
                raise ValueError('news_price_component_path_invalid')
            content = _frozen_bytes(origin / 'prices' / name, digest, limit=96_000_000)
            (stage / 'prices' / name).write_bytes(content)
            files['prices/' + name] = digest
        for name, content in [('request.json', request_raw), ('news.json', news_raw),
                              ('report.json', json.dumps(report, default=lambda v: v.isoformat(), sort_keys=True, allow_nan=False).encode())]:
            (stage / name).write_bytes(content)
            files[name] = hashlib.sha256(content).hexdigest()
        engine_code_digest.cache_clear()
        if engine_code_digest() != engine:
            raise ValueError('news_engine_changed_during_run')
        completion = {'contractVersion': 'news-equity-study-completion-v1', 'files': files,
                      'qualificationIssued': False, 'executionAuthorized': False}
        args.output.mkdir(mode=0o700, parents=True, exist_ok=False)
        for name in files:
            target = args.output / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(stage / name, target)
        (args.output / 'completed.json').write_text(json.dumps(completion, sort_keys=True))
    return {'status': report['status'], 'path': str(args.output), 'qualificationIssued': False, 'executionAuthorized': False}


def add_news_event_commands(subparsers):
    parser = subparsers.add_parser('news-equity-study', help='Study received news using frozen US-equity returns.')
    parser.add_argument('--request', required=True, type=Path)
    parser.add_argument('--request-sha256', required=True)
    parser.add_argument('--output', required=True, type=Path)
    parser.set_defaults(func=news_equity_command)
