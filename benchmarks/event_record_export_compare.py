"""Compare event report record materialization and streaming record export."""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.event_engine_compare import MultiOrderStrategy, make_fusion_bars  # noqa: E402
from cortexbacktest import BacktestEngine, ColumnarBarFeed, EventRecordSink  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare event record export modes.")
    parser.add_argument("--bars", type=int, default=1_000)
    parser.add_argument("--symbols", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    args = parser.parse_args()

    result = BacktestEngine(snapshot_frequency="bar").run(
        ColumnarBarFeed.from_bars(make_fusion_bars(args.bars, args.symbols)),
        MultiOrderStrategy(),
    )
    print("mode,seconds,orders,fills,equity_points")
    modes = [
        ("json_report_records", lambda path: result.report(include_records=True)),
        ("stream_csv_records", lambda path: result.records.write_dir(path / "csv", format="csv")),
        ("stream_ndjson_records", lambda path: result.records.write_dir(path / "ndjson", format="ndjson")),
    ]
    for mode, fn in modes:
        seconds, _ = _best_runtime(fn, repeat=args.repeat, warmup=args.warmup)
        print(f"{mode},{seconds:.6f},{len(result.orders)},{len(result.fills)},{result.performance.equity_curve().height}")
    runtime_seconds, runtime_result = _best_runtime(
        lambda path: BacktestEngine(
            snapshot_frequency="bar",
            result_retention="summary",
            record_sink=EventRecordSink(path / "runtime_ndjson", format="ndjson"),
        ).run(
            ColumnarBarFeed.from_bars(make_fusion_bars(args.bars, args.symbols)),
            MultiOrderStrategy(),
        ),
        repeat=args.repeat,
        warmup=args.warmup,
    )
    print(
        "run_stream_ndjson_summary,"
        f"{runtime_seconds:.6f},{runtime_result.order_record_count},{runtime_result.fill_count},"
        f"{runtime_result.performance.equity_curve().height}"
    )
    return 0


def _best_runtime(fn, *, repeat: int, warmup: int):
    for _ in range(warmup):
        tmp = Path(tempfile.mkdtemp(prefix="fusion-record-bench-"))
        try:
            fn(tmp)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    best = float("inf")
    best_result = None
    for _ in range(repeat):
        tmp = Path(tempfile.mkdtemp(prefix="fusion-record-bench-"))
        start = time.perf_counter()
        try:
            result = fn(tmp)
            elapsed = time.perf_counter() - start
            if elapsed < best:
                best = elapsed
                best_result = result
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    assert best_result is not None
    return best, best_result


if __name__ == "__main__":
    raise SystemExit(main())
