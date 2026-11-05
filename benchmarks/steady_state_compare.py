"""Steady-state throughput of CortexBacktest's measurement modes.

Build the payload, run once to warm up, then time the run only, best of N.
Other engines are compared on real data in `benchmarks/real_data_comparison/`.

    python benchmarks/steady_state_compare.py --bars 100000 --output results.csv
"""

from __future__ import annotations

import argparse
import csv
import platform
import sys
import time
from functools import partial
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.event_engine_compare import BuyFirstSlice, make_fusion_bars  # noqa: E402


def _time_best(setup: Callable[[], Any], run: Callable[[Any], Any], repeats: int) -> float:
    run(setup())  # warm up imports, JIT, and any lazy engine state
    best = float("inf")
    for _ in range(repeats):
        payload = setup()
        start = time.perf_counter()
        run(payload)
        best = min(best, time.perf_counter() - start)
    return best


def _fusion(
    bars: int, *, throughput: bool = False, prebuilt: bool = False,
) -> tuple[Callable[[], Any], Callable[[Any], Any]]:
    """Build one of the four existing Cortex measurement modes.

    Raw bars pay sorting/validation/grouping inside the timed run. Prebuilt
    feeds exclude ingestion and are reported separately. Throughput mode also
    omits per-bar retained output, so it is not a full-statistics run.
    """
    from cortexbacktest import BacktestEngine
    from cortexbacktest.broker import BrokerSimulator
    from cortexbacktest.event.feed import ColumnarBarFeed

    options = {
        "snapshot_frequency": "none", "result_retention": "summary", "build_manifest": False,
    } if throughput else {}

    def setup() -> Any:
        data = make_fusion_bars(bars, 1, volume=1_000_000)
        return ColumnarBarFeed.from_bars(data) if prebuilt else data

    def run(payload: Any) -> None:
        engine = BacktestEngine(
            broker=BrokerSimulator(max_volume_share=1.0), **options,
        )
        engine.run(payload, BuyFirstSlice())

    return setup, run


ENGINES = {
    "fusion": _fusion,
    "fusion (throughput mode)": partial(_fusion, throughput=True),
    # Reported last and separately: not like-for-like rows (see their docstrings).
    "fusion (prebuilt feed)": partial(_fusion, prebuilt=True),
    "fusion (throughput + prebuilt feed)": partial(_fusion, throughput=True, prebuilt=True),
}

NOT_COMPARABLE = {"fusion (prebuilt feed)", "fusion (throughput + prebuilt feed)"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bars", type=int, default=100_000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    print(f"steady state, {args.bars:,} bars, best of {args.repeats}, data construction excluded")
    print(f"{platform.python_implementation()} {platform.python_version()} on {platform.machine()}\n")

    rows = []
    for name, factory in ENGINES.items():
        try:
            setup, run = factory(args.bars)
            seconds = _time_best(setup, run, args.repeats)
        except Exception as error:  # one failing mode must not abort the sweep
            print(f"{name:<24} unavailable: {type(error).__name__}: {error}")
            rows.append({"engine": name, "bars": args.bars, "seconds": "", "bars_per_second": "",
                         "available": False, "comparable": name not in NOT_COMPARABLE, "reason": f"{type(error).__name__}: {error}"})
            continue
        rate = args.bars / seconds
        note = "  <- not comparable: ingestion hoisted out of the timed run" if name in NOT_COMPARABLE else ""
        print(f"{name:<24} {seconds:8.4f}s  {rate:>12,.0f} bars/s{note}")
        rows.append({"engine": name, "bars": args.bars, "seconds": f"{seconds:.6f}",
                     "bars_per_second": f"{rate:.1f}", "available": True, "comparable": name not in NOT_COMPARABLE, "reason": ""})

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["engine", "bars", "seconds", "bars_per_second", "available", "comparable", "reason"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
