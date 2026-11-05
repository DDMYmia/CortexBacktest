"""Throughput and peak memory of the option-chain ingest path, stage by stage.

Synthetic chains shaped like a real EOD file (dates x expirations x strikes x
call/put), so the row counts and the column mix match what a Kaggle year or a
DoltHub pull actually produces. No market data and no network.

    python benchmarks/option_chain_ingest.py --rows 400000 \
        --output benchmark-results/option-chain-ingest/latest.json

Stages, in the order a caller runs them:

- **normalize** — vendor table to the canonical schema (`normalize_option_chain`)
- **validate** — structural errors and quote-defect warnings
- **clean** — per-rule filtering
- **store_write** / **store_scan** — hive-partitioned Parquet round trip
- **contracts** / **expiry_events** — the two per-contract stages, which are
  O(distinct contracts) rather than O(rows)
- **frames** — the adapter to `OptionMarketFrame`, which is the only stage that
  must materialise one Python object per quote and therefore the one that
  dominates at scale

`peak_rss_mb` is this process's high-water mark, `peak_rss_over_import_mb`
subtracts the RSS measured after imports, the same convention as
`extreme_scaling_compare.py`.
"""

from __future__ import annotations

import argparse
import json
import resource
import shutil
import sys
import tempfile
from datetime import date, timedelta
from pathlib import Path
from statistics import median
from time import perf_counter

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cortexbacktest.options.chains import (  # noqa: E402
    ChainFilter, OptionChainStore, clean_option_chain, expiry_events_from_underlying,
    normalize_option_chain, option_chain_to_frames, option_contracts_from_chain,
    validate_option_chain,
)


def _peak_rss_mb() -> float:
    """ru_maxrss is bytes on macOS and kilobytes on Linux."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024 * 1024) if sys.platform == "darwin" else peak / 1024


def synthetic_wide_chain(rows: int, *, strikes: int = 40, expirations: int = 8) -> pl.DataFrame:
    """Build a wide fixture; requested normalized rows round to whole date grids."""
    pairs = max(1, rows // 2)
    per_date = strikes * expirations
    dates = max(1, pairs // per_date)
    quote_dates, expiry_dates, strike_values = [], [], []
    for day in range(dates):
        quote = date(2020, 1, 2) + timedelta(days=day)
        for expiry in range(expirations):
            for strike in range(strikes):
                quote_dates.append(quote.isoformat())
                expiry_dates.append((quote + timedelta(days=7 * (expiry + 1))).isoformat())
                strike_values.append(300.0 + strike)
    count = len(quote_dates)
    # Built with expressions rather than per-row Python: the fixture must not
    # be what the peak-memory number is measuring.
    strikes_column = pl.Series("strike", strike_values)
    intrinsic_call = (320.0 - strikes_column).clip(lower_bound=0.05)
    intrinsic_put = (strikes_column - 320.0).clip(lower_bound=0.05)
    return pl.DataFrame({
        "[QUOTE_DATE]": quote_dates, "[EXPIRE_DATE]": expiry_dates,
        "[UNDERLYING_LAST]": pl.Series([320.0] * count), "[STRIKE]": strikes_column,
        "[C_BID]": intrinsic_call + 1.0, "[C_ASK]": intrinsic_call + 1.1,
        "[C_IV]": pl.Series([0.18] * count), "[C_DELTA]": pl.Series([0.5] * count),
        "[C_GAMMA]": pl.Series([0.02] * count), "[C_VEGA]": pl.Series([0.3] * count),
        "[C_THETA]": pl.Series([-0.05] * count), "[C_RHO]": pl.Series([0.01] * count),
        "[C_VOLUME]": pl.Series([100.0] * count),
        "[P_BID]": intrinsic_put + 1.0, "[P_ASK]": intrinsic_put + 1.1,
        "[P_IV]": pl.Series([0.20] * count), "[P_DELTA]": pl.Series([-0.5] * count),
        "[P_GAMMA]": pl.Series([0.02] * count), "[P_VEGA]": pl.Series([0.3] * count),
        "[P_THETA]": pl.Series([-0.05] * count), "[P_RHO]": pl.Series([-0.01] * count),
        "[P_VOLUME]": pl.Series([100.0] * count),
    })


def _timed(label: str, repeats: int, call):
    times = []
    value = None
    for _ in range(repeats):
        before = perf_counter()
        value = call()
        times.append(perf_counter() - before)
    return label, median(times), value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=200_000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--frame-rows", type=int, default=100_000,
                        help="rows fed to the OptionMarketFrame adapter (it is O(quotes) "
                             "in Python objects, so it is measured at its own size)")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    baseline_rss = _peak_rss_mb()
    raw = synthetic_wide_chain(args.rows)
    stages: dict[str, float] = {}
    # Process high-water marks include retained inputs and allocator history;
    # differences do not isolate a stage's allocations or peak memory.
    watermarks: dict[str, float] = {"after_fixture": _peak_rss_mb()}

    _, seconds, chain = _timed("normalize", args.repeats,
                               lambda: normalize_option_chain(raw, "kaggle_spy_eod"))
    stages["normalize"] = seconds
    assert chain is not None
    rows = chain.height
    watermarks["after_normalize"] = _peak_rss_mb()

    stages["validate"] = _timed("validate", args.repeats,
                                lambda: validate_option_chain(chain))[1]
    stages["clean"] = _timed("clean", args.repeats,
                             lambda: clean_option_chain(chain, ChainFilter()))[1]
    watermarks["after_clean"] = _peak_rss_mb()

    directory = Path(tempfile.mkdtemp(prefix="chain-bench-"))
    try:
        store = OptionChainStore(directory)
        before = perf_counter()
        store.write(chain)
        stages["store_write"] = perf_counter() - before
        stages["store_scan"] = _timed("store_scan", args.repeats,
                                      lambda: store.scan().collect())[1]
    finally:
        shutil.rmtree(directory, ignore_errors=True)

    watermarks["after_store"] = _peak_rss_mb()

    contracts = option_contracts_from_chain(chain)
    stages["contracts"] = _timed("contracts", args.repeats,
                                 lambda: option_contracts_from_chain(chain))[1]
    stages["expiry_events"] = _timed(
        "expiry_events", args.repeats,
        lambda: expiry_events_from_underlying(chain, contracts))[1]

    slice_rows = min(args.frame_rows, rows)
    subset = chain.head(slice_rows)
    stages["frames"] = _timed("frames", args.repeats,
                              lambda: option_chain_to_frames(subset))[1]
    watermarks["after_frames"] = _peak_rss_mb()

    peak = _peak_rss_mb()
    report = {
        "rows": rows,
        "frame_rows": slice_rows,
        "repeats": args.repeats,
        "seconds": stages,
        "rows_per_second": {name: (slice_rows if name == "frames" else rows) / seconds
                            for name, seconds in stages.items() if seconds > 0},
        "peak_rss_after_stage_mb": watermarks,
        "peak_rss_mb": peak,
        "peak_rss_over_import_mb": peak - baseline_rss,
        "polars": pl.__version__,
        "threads": pl.thread_pool_size(),
    }
    text = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
