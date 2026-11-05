"""Isolated end-to-end Parquet replay; synthetic input generation is untimed.

Run each size in a fresh process to compare peak RSS. A 1,000-symbol universe
is emitted in minute-major order, without creating the full input in memory.
"""
from argparse import ArgumentParser
from datetime import UTC, datetime, timedelta
import json
from pathlib import Path
import resource
import sys
from tempfile import TemporaryDirectory
from time import perf_counter

import polars as pl
import pyarrow.parquet as pq

from cortexbacktest import BacktestEngine, BrokerSimulator, Order, ParquetBarFeed, Side, Strategy


class NoOp(Strategy):
    def on_bars(self, bars):
        return []


class BuyHold(Strategy):
    def __init__(self):
        self.bought = False

    def on_bars(self, bars):
        if self.bought:
            return []
        self.bought = True
        return [Order(bar.symbol, bar.timestamp, Side.BUY, 1) for bar in bars]


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--minutes', type=int, default=1000)
    parser.add_argument('--symbols', type=int, default=1000)
    parser.add_argument('--batch-rows', type=int, default=65536)
    parser.add_argument('--strategy', choices=['noop', 'buy-hold'], default='noop')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if min(args.minutes, args.symbols, args.batch_rows) < 1:
        parser.error('sizes must be positive')
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / 'bars.parquet'
        writer = None
        try:
            for start in range(0, args.minutes, 50):
                count = min(50, args.minutes - start)
                dates = [datetime(2020, 1, 1, tzinfo=UTC) + timedelta(minutes=i)
                         for i in range(start, start + count) for _ in range(args.symbols)]
                frame = pl.DataFrame({'date': dates,
                    'symbol': [f'S{i:04}' for i in range(args.symbols)] * count,
                    **{name: [100.0] * len(dates) for name in ('open', 'high', 'low', 'close', 'volume')}})
                table = frame.to_arrow()
                if writer is None:
                    writer = pq.ParquetWriter(path, table.schema)
                writer.write_table(table)
            del frame, table, dates
        finally:
            if writer is not None:
                writer.close()
        before = perf_counter()
        broker = BrokerSimulator(initial_cash=1_000_000)
        result = BacktestEngine(broker=broker, snapshot_frequency='none', result_retention='summary',
                                build_manifest=False).run(
            ParquetBarFeed(path, batch_rows=args.batch_rows), NoOp() if args.strategy == 'noop' else BuyHold())
        elapsed = perf_counter() - before
        payload = {'strategy': args.strategy, 'minutes': args.minutes, 'symbols': args.symbols,
            'observations': args.minutes * args.symbols, 'seconds': elapsed,
            'observations_per_second': args.minutes * args.symbols / elapsed,
            'peak_rss_mib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss /
                            (1024**2 if sys.platform == 'darwin' else 1024),
            'batch_rows': args.batch_rows, 'fills': result.fill_count,
            'ledger_equity': broker.state.equity(), 'positions': len(broker.state.positions),
            'method': 'Selected strategy; disk decode, validation, materialization and replay included; input generation excluded from timing, included in process RSS; no manifest or snapshots.'}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + '\n')
        print(json.dumps(payload))


if __name__ == '__main__':
    main()
