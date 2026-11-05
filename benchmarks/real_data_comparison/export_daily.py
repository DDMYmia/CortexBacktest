"""Build daily OHLCV from the 1-minute research layer (run on the data server).

    python export_daily.py --root /path/to/market-data --out recent_daily.parquet

Daily bars are the first open, max high, min low, last close and summed volume of
each ticker's minute bars on a date. Only (ticker, date) pairs that the census
daily store also has are kept: the minute layer labels some extended-hours bars
with a weekend date. The census close agrees with the aggregated close
(median relative difference 0 on 2024-01..2026-03). The store is unadjusted.
"""
from __future__ import annotations

import argparse
from datetime import date

import polars as pl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="Market data root on the data server.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--start", type=date.fromisoformat, default=date(2024, 1, 2))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2026, 3, 31))
    parser.add_argument("--tickers", type=int, default=100, help="Most liquid tickers to keep.")
    args = parser.parse_args()

    census = pl.read_parquet(f"{args.root}/census/daily_bars.parquet").filter(
        pl.col("day").is_between(args.start, args.end))
    sessions = census["day"].n_unique()
    liquid = (census.group_by("ticker")
              .agg(pl.len().alias("n"), (pl.col("close") * pl.col("volume")).mean().alias("dollar_volume"))
              .filter(pl.col("n") >= 0.95 * sessions)
              .sort("dollar_volume", descending=True)
              .head(args.tickers)["ticker"].to_list())
    minutes = pl.concat([
        pl.read_parquet(f"{args.root}/research/equities_1m/year={year}/*.parquet")
        .filter(pl.col("ticker").is_in(liquid))
        for year in range(args.start.year, args.end.year + 1)
    ]).with_columns(pl.col("timestamp").dt.date().alias("day")).sort(["ticker", "timestamp"])
    daily = (minutes.group_by(["ticker", "day"], maintain_order=True)
             .agg(pl.col("open").first(), pl.col("high").max(), pl.col("low").min(),
                  pl.col("close").last(), pl.col("volume").sum())
             .join(census.select("ticker", "day"), on=["ticker", "day"], how="semi")
             .sort(["ticker", "day"]))
    daily.write_parquet(args.out)
    print(f"wrote {args.out}: {daily.height} rows, {daily['ticker'].n_unique()} tickers, "
          f"{daily['day'].n_unique()} sessions")


if __name__ == "__main__":
    main()
