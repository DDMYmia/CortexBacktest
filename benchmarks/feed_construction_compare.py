"""Time building a `ColumnarBarFeed` from a Polars frame vs. from `list[Bar]`.

`from_polars` is the path a research workflow actually uses (a wide
date x symbol price frame is the natural output of the data layer); `from_bars`
is what every other benchmark in this directory uses to build its payload, so
it is the thing regressions elsewhere are usually already measured against.
Timing both, at a fixed size, on every run gives the perf gate a number for
"how expensive is turning a Polars frame into an event feed" that nothing else
here covers.

    python benchmarks/feed_construction_compare.py --dates 500 --symbols 200

Construction only: this does not run an engine over the resulting feed.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cortexbacktest import ColumnarBarFeed  # noqa: E402

from benchmarks.event_engine_compare import make_fusion_bars  # noqa: E402


def _wide_price_frame(dates: int, symbols: int) -> pl.DataFrame:
    date_values = pl.date_range(
        start=date(2024, 1, 2),
        end=date(2024, 1, 2) + timedelta(days=dates - 1),
        interval="1d",
        eager=True,
    )
    columns: dict[str, object] = {"date": date_values}
    for idx in range(symbols):
        base = 100.0 + idx
        columns[f"S{idx:04d}"] = [base + day * 0.01 for day in range(dates)]
    wide = pl.DataFrame(columns)
    long_frames = [
        wide.select(
            "date",
            pl.lit(column).alias("symbol"),
            pl.col(column).alias("open"),
            pl.col(column).alias("high"),
            pl.col(column).alias("low"),
            pl.col(column).alias("close"),
        )
        for column in wide.columns
        if column != "date"
    ]
    return pl.concat(long_frames, how="vertical")


def _time_best(fn, *, repeat: int) -> float:
    fn()  # warm up
    best = float("inf")
    for _ in range(repeat):
        started = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - started)
    return best


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dates", type=int, default=500)
    parser.add_argument("--symbols", type=int, default=200)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    frame = _wide_price_frame(args.dates, args.symbols)
    row_count = frame.height
    bars = make_fusion_bars(args.dates, args.symbols)

    rows = []
    for source, fn in (
        ("from_polars", lambda: ColumnarBarFeed.from_polars(frame)),
        ("from_bars", lambda: ColumnarBarFeed.from_bars(bars)),
    ):
        seconds = _time_best(fn, repeat=args.repeat)
        rows.append((source, row_count, seconds, row_count / seconds))

    print("source,rows,seconds,rows_per_second")
    for source, count, seconds, rate in rows:
        print(f"{source},{count},{seconds:.6f},{rate:.1f}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            "source,rows,seconds,rows_per_second\n"
            + "".join(f"{s},{c},{sec:.6f},{r:.1f}\n" for s, c, sec, r in rows)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
