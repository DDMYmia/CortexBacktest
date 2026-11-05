"""Build a split table for an unadjusted minute store, from two half-sources.

Neither source is sufficient alone, and they fail in opposite directions:

- **The bars know *when*, not *how much*.** A split day is unmistakable in
  unadjusted prices, but the ratio is not recoverable from them. Measured on
  three known splits: the price-implied factor missed by up to 14% (TSLA rose
  12.6% the same day it split 5:1, landing at 4.31), and 1/4 and 1/5 are only
  25% apart -- so price alone would have labelled that 5:1 split a 4:1. Volume
  is worse (NVDA's split-day volume ratio was 2.32 against a true factor of 4).
  Calling a 5:1 a 4:1 is worse than reporting nothing, because every adjusted
  series downstream inherits the error silently.

- **yfinance knows *how much*, but only for the living.** It returns the exact
  ratios for AAPL, TSLA and NVDA, and nothing at all for SIVB, TWTR, ABMD or
  FRC -- the delisted symbols that are the whole reason this store is worth
  keeping.

The obvious escape -- that a 1-for-20 reverse split moves the price so far that
the stock's own return stops mattering -- was tested on 630 confirmed ratios and
does not hold. No factor band's worst error stays under half the distance to the
nearest ratio that actually occurs; the median error is 1.5-2.6% everywhere but
the tail reaches 17-27%, and the 38 observed ratios are packed too closely
(0.125, 0.167, 0.2 are neighbours, so are 38, 40, 50) for that to be survivable.
Snapping to the nearest observed ratio is right 85.6% of the time overall and
96.6% on reverse splits, which is a number for a human deciding one symbol, not
a licence to adjust a store automatically.

So the bars enumerate candidate ex-dates across every symbol including the
delisted ones, and yfinance supplies the authoritative ratio wherever it can.
Rows it cannot reach keep the price-implied hint and are marked
``verified=False``: a candidate to check, never a fact to adjust by.

    python derive_corporate_actions.py <bars dir> [out.parquet] [--no-yfinance]
"""
from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

# 5:4 splits exist, but so do -20% earnings days, and at that size the two are
# the same shape. The floor is where the detector stops guessing; smaller splits
# are missed rather than invented.
MIN_FACTOR = 1.4
# Sub-dollar stocks on the pre-2001 fractional grid produce ratios like 1.25
# from a single tick. A price floor removes that whole class of false positive.
MIN_PRICE = 5.0
# Normal day-over-day volume ratios sit inside roughly [0.6, 1.7]; a split moves
# share count by the factor itself, so anything beyond the band corroborates.
VOLUME_BAND = (0.6, 1.7)
MAX_SESSION_GAP_DAYS = 5


def daily_bars(root: Path) -> pl.DataFrame:
    """Last close and total share volume per symbol-session, in exchange time."""
    files = sorted(root.glob("ohlcv_*.parquet"))
    if not files:
        raise SystemExit(f"no ohlcv_*.parquet under {root}")
    print(f"scanning {len(files)} files", flush=True)
    et = pl.col("timestamp").dt.convert_time_zone("America/New_York")
    return (
        pl.scan_parquet(files)
        .with_columns(et.dt.date().alias("day"))
        .sort("timestamp")
        .group_by(["ticker", "day"])
        .agg(pl.col("close").last().alias("close"), pl.col("volume").sum().alias("volume"))
        .sort(["ticker", "day"])
        .collect()
    )


def candidate_dates(daily: pl.DataFrame) -> pl.DataFrame:
    moves = daily.with_columns(
        pl.col("close").shift(1).over("ticker").alias("prev_close"),
        (pl.col("volume") / pl.col("volume").shift(1)).over("ticker").alias("volume_ratio"),
        pl.col("day").shift(1).over("ticker").alias("prev_day"),
    ).drop_nulls("prev_close")
    moves = moves.with_columns((pl.col("prev_close") / pl.col("close")).alias("implied_factor"))
    moves = moves.filter(
        # Adjacent sessions only: a gap across missing months would read months
        # of drift as one overnight move.
        ((pl.col("day") - pl.col("prev_day")).dt.total_days() <= MAX_SESSION_GAP_DAYS)
        & (pl.col("prev_close") >= MIN_PRICE)
        & ((pl.col("implied_factor") >= MIN_FACTOR) | (pl.col("implied_factor") <= 1 / MIN_FACTOR))
    )
    # No snapping to a "nearest simple ratio": on the three known splits that
    # produced 3.875, 4.111 and 4.286 for true factors of 4, 4 and 5 -- false
    # precision that reads as an answer. The raw implied factor says the same
    # thing without pretending.
    forward = pl.col("implied_factor") >= 1
    return (
        moves.filter(
            pl.when(forward)
            .then(pl.col("volume_ratio") > VOLUME_BAND[1])
            .otherwise(pl.col("volume_ratio") < VOLUME_BAND[0])
            .fill_null(False)
        )
        .select(
            pl.col("ticker").alias("symbol"), "ex_date" if "ex_date" in moves.columns else pl.col("day").alias("ex_date"),
            "implied_factor", "volume_ratio",
        )
        .sort(["symbol", "ex_date"])
    )


def verify(candidates: pl.DataFrame) -> pl.DataFrame:
    """Replace the hint with the vendor's ratio wherever the vendor has one."""
    try:
        import yfinance as yf
    except ImportError:
        print("yfinance not installed; every row stays unverified", flush=True)
        return candidates.with_columns(pl.col("implied_factor").alias("ratio"),
                                       pl.lit(False).alias("verified"))
    symbols = candidates["symbol"].unique().sort().to_list()
    print(f"asking yfinance about {len(symbols):,} symbols", flush=True)
    known: dict[tuple[str, object], float] = {}
    for index, symbol in enumerate(symbols, 1):
        if index % 200 == 0:
            print(f"  {index:,}/{len(symbols):,}", flush=True)
        try:
            splits = yf.Ticker(symbol).splits
        except Exception:
            continue
        if splits is None or len(splits) == 0:
            continue
        for stamp, ratio in splits.items():
            known[(symbol, stamp.date())] = float(ratio)
    return candidates.with_columns(
        pl.struct(["symbol", "ex_date"]).map_elements(
            lambda row: known.get((row["symbol"], row["ex_date"])), return_dtype=pl.Float64
        ).alias("vendor_ratio")
    ).with_columns(
        pl.coalesce("vendor_ratio", "implied_factor").alias("ratio"),
        pl.col("vendor_ratio").is_not_null().alias("verified"),
    )


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    root = Path(args[0])
    out = Path(args[1]) if len(args) > 1 else None
    candidates = candidate_dates(daily_bars(root))
    if candidates.is_empty():
        print("no split candidates")
        return
    print(f"candidate ex-dates: {candidates.height:,} across {candidates['symbol'].n_unique():,} symbols")
    table = (candidates.with_columns(pl.col("implied_factor").alias("ratio"),
                                     pl.lit(False).alias("verified"))
             if "--no-yfinance" in sys.argv else verify(candidates))
    if "verified" in table.columns:
        verified = int(table["verified"].sum())
        print(f"verified by vendor: {verified:,} / {table.height:,} ({verified / table.height:.1%})")
        print("\nunverified lack a matched vendor ratio (missing history or failed query); "
              "their ratio is the price-implied hint and must not be adjusted by blindly.")
    print(table.head(15))
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        table.write_parquet(out)
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
