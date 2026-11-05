"""Source-documented strategy ladder over identical synthetic minute bars.

End-to-end times include Parquet validation, hashing, strategy, broker and
streamed statistics. Input generation and optional cProfile passes are excluded.
No claims about investment performance or real-market capacity are made.
"""
from argparse import ArgumentParser
import cProfile
from datetime import UTC, datetime, timedelta
import json
from pathlib import Path
import platform
import pstats
import resource
import sys
from time import perf_counter

import numpy as np
import polars as pl
import pyarrow.parquet as pq

from cortexbacktest import BacktestEngine, BrokerSimulator, ParquetBarFeed
from cortexbacktest.event.analyzers import StreamingStatsAnalyzer
from cortexbacktest.event.record_sink import EventRecordSink
from cortexbacktest.strategy_library import STRATEGIES


def write_input(path, minutes, symbols, seed):
    rng = np.random.default_rng(seed)
    previous = np.full(symbols, 100.0)
    names = [f'S{i:04}' for i in range(symbols)]
    writer = None
    try:
        for start in range(0, minutes, 50):
            size = min(50, minutes - start)
            changes = rng.normal(0.00002, .004, (size, symbols))
            close = previous * np.exp(np.cumsum(changes, axis=0))
            opens = np.vstack((previous, close[:-1]))
            frame = pl.DataFrame({
                'date': [datetime(2024, 1, 2, 9, 30, tzinfo=UTC) + timedelta(minutes=i)
                         for i in range(start, start + size) for _ in names],
                'symbol': names * size, 'open': opens.ravel(), 'close': close.ravel(),
                'high': (np.maximum(opens, close) * 1.001).ravel(),
                'low': (np.minimum(opens, close) * .999).ravel(),
                'volume': np.full(size * symbols, 12.0),
            })
            table = frame.to_arrow()
            if writer is None:
                writer = pq.ParquetWriter(path, table.schema)
            writer.write_table(table)
            previous = close[-1]
    finally:
        if writer is not None:
            writer.close()


def execute(path, name, mode, sink_path=None):
    analyzer = StreamingStatsAnalyzer()
    sink = EventRecordSink(sink_path, format='parquet', buffer_rows=2000) if sink_path else None
    broker = BrokerSimulator(initial_cash=100_000_000, commission_bps=1, slippage_bps=1,
                             max_volume_share=.25)
    try:
        result = BacktestEngine(broker=broker, trade_on_close=False, build_manifest=True,
            data_validation=mode, result_retention='summary', snapshot_frequency='final',
            analyzers=[analyzer], record_sink=sink).run(ParquetBarFeed(path), STRATEGIES[name](quantity=10))
    finally:
        if sink is not None:
            sink.close()
    summary = result.summary()
    # A final-only equity curve cannot supply time-series performance metrics.
    # Return/drawdown statistics come from the complete online accumulator.
    summary = {key: summary[key] for key in ('final_equity', 'fills', 'orders', 'order_records')}
    return {'summary': summary, 'stats': result.analyzer_results['streaming_stats'],
            'manifest': result.run_manifest, 'open_orders': result.open_order_count,
            'retained_broker_fills': len(broker.fills), 'retained_broker_orders': len(broker.order_records)}


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--minutes', type=int, default=600)
    parser.add_argument('--symbols', type=int, default=200)
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--strategies', default=','.join(STRATEGIES))
    parser.add_argument('--validation', choices=['both', 'preflight', 'stream'], default='both')
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--sink', action='store_true')
    args = parser.parse_args()
    if min(args.minutes, args.symbols, args.repeat) < 1:
        parser.error('sizes and repeat must be positive')
    names = args.strategies.split(',')
    if set(names) - STRATEGIES.keys():
        parser.error('unknown strategies')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / 'input.parquet'
    write_input(path, args.minutes, args.symbols, args.seed)
    modes = ['preflight', 'stream'] if args.validation == 'both' else [args.validation]
    results = []
    for name in names:
        reference = None
        for mode in modes:
            for repeat in range(args.repeat):
                started = perf_counter()
                payload = execute(path, name, mode,
                    args.output_dir / f'{name}-{mode}-{repeat}-records' if args.sink else None)
                seconds = perf_counter() - started
                identity = {key: payload[key] for key in ('summary', 'stats', 'manifest', 'open_orders')}
                if reference is None:
                    reference = identity
                assert identity == reference, 'validation modes/repeats changed results'
                assert payload['retained_broker_fills'] == payload['retained_broker_orders'] == 0
                row = {'strategy': name, 'validation': mode, 'repeat': repeat, 'seconds': seconds,
                    'observations_per_second': args.minutes * args.symbols / seconds, **payload}
                results.append(row)
                (args.output_dir / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
                print(json.dumps({key: row[key] for key in ('strategy', 'validation', 'seconds', 'summary')}), flush=True)
        if args.profile:
            profiler = cProfile.Profile()
            profiler.runcall(execute, path, name, 'stream')
            with (args.output_dir / f'{name}.profile.txt').open('w') as handle:
                pstats.Stats(profiler, stream=handle).strip_dirs().sort_stats('cumtime').print_stats(35)
    (args.output_dir / 'environment.json').write_text(json.dumps({
        'python': platform.python_version(), 'platform': platform.platform(),
        'minutes': args.minutes, 'symbols': args.symbols, 'seed': args.seed,
        'sink': args.sink, 'repeat': args.repeat,
        'process_peak_rss_mib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss /
            (1024**2 if sys.platform == 'darwin' else 1024),
        'rss_scope': 'Whole benchmark process high-water mark; not per strategy or mode',
    }, indent=2) + '\n')


if __name__ == '__main__':
    main()
