#!/usr/bin/env python
"""Pull a public option chain into the engine's Parquet store.

Two sources, because between them they cover both axes a free options study
needs — DoltHub for breadth (thousands of US underlyings, ~2019 onwards) and a
local Kaggle/OptionsDX file for depth (SPY back to 2010):

    # DoltHub post-no-preference/options over its public SQL API (no key)
    python scripts/fetch_option_chains.py dolthub --underlying SPY \
        --start 2024-01-02 --end 2024-01-31 --root data/options

    # A file already on disk: Kaggle SPY EOD parquet, or an OptionsDX month
    python scripts/fetch_option_chains.py file data/raw/spy_eod_2020.parquet \
        --source kaggle_spy_eod --root data/options
    python scripts/fetch_option_chains.py file data/raw/spy_2020_january.txt \
        --source optionsdx --underlying SPY --root data/options

Getting the raw files in the first place is the vendors' business, not this
script's: `kaggle datasets download -d <slug> --unzip`, or an OptionsDX download.

Everything lands normalised, validated and cleaned in
`<root>/underlying=<TICKER>/year=<YYYY>/`. Nothing here is a runtime dependency
of the package: it is stdlib plus the engine itself, and it is deliberately not
imported from `src/`.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen

import polars as pl

from cortexbacktest.options.chains import (
    ChainFilter, ChainTimePolicy, OptionChainStore, clean_option_chain, iter_normalized_chunks,
    normalize_option_chain, validate_option_chain, with_underlying_prices,
)

DOLTHUB_API = "https://www.dolthub.com/api/v1alpha1/post-no-preference/options"
# The same publisher's stock repository: unadjusted daily closes and splits.
DOLTHUB_STOCKS_API = "https://www.dolthub.com/api/v1alpha1/post-no-preference/stocks"
# The repository's default branch is `master`, not `main`; a wrong ref fails with
# "branch not found" rather than an empty result.
DOLTHUB_BRANCH = "master"
DOLTHUB_COLUMNS = ("date", "act_symbol", "expiration", "strike", "call_put",
                   "bid", "ask", "vol", "delta", "gamma", "theta", "vega", "rho")
# Verified against the live API: a response carries at most 1000 rows, and one
# that hit the cap reports status `RowLimit` instead of `Success`.
DOLTHUB_ROW_CAP = 1000


def _dolthub_query(sql: str, branch: str, *, attempts: int = 3,
                   api: str = DOLTHUB_API) -> list[dict[str, object]]:
    """One API call, retried on the server's query deadline.

    A response that hit the row cap reports `RowLimit` rather than `Success`;
    that is a truncated page, not a failure. `context deadline exceeded` is the
    server giving up on a query, which retrying sometimes clears once the
    relevant chunks are warm.
    """
    for attempt in range(1, attempts + 1):
        with urlopen(f"{api}/{branch}?q={quote(sql)}", timeout=180) as response:
            payload = json.loads(response.read())
        status = payload.get("query_execution_status")
        if status in ("Success", "RowLimit"):
            return payload.get("rows", [])
        message = str(payload.get("query_execution_message"))
        if "deadline" not in message or attempt == attempts:
            raise RuntimeError(f"DoltHub query failed: {message}")
        print(f"  retrying after: {message}", flush=True)
    return []


def dolthub_rows(underlying: str, start: date, end: date, *, page: int = DOLTHUB_ROW_CAP,
                 max_rows: int | None = None, branch: str = DOLTHUB_BRANCH) -> pl.DataFrame:
    """Fetch one calendar date per request, paging within a date if needed.

    Deliberately not one query for the whole range: the table's primary key
    starts with `date`, so `date = '...'` is an index seek that answers in about
    a second, while `date BETWEEN ... AND act_symbol = ...` scans every symbol on
    every date in the window and exceeds the server's query deadline — measured,
    not assumed. Dates with no rows (weekends, holidays, symbols that were not
    covered yet) come back empty and cost about the same as a cache hit.

    Every value arrives as a JSON string, decimals included; the chain
    normaliser casts them, so the frame is handed over as read.
    """
    if not re.fullmatch(r"[A-Za-z0-9.\-]{1,10}", underlying):
        raise ValueError(f"implausible ticker {underlying!r}")
    if start > end:
        raise ValueError("start is after end")
    if page <= 0:
        raise ValueError("page must be positive")
    collected: list[dict[str, object]] = []
    day = start
    while day <= end and (max_rows is None or len(collected) < max_rows):
        before_day = len(collected)
        offset = 0
        while max_rows is None or len(collected) < max_rows:
            limit = min(page, DOLTHUB_ROW_CAP)
            if max_rows is not None:
                limit = min(limit, max_rows - len(collected))
            rows = _dolthub_query(
                f"SELECT {', '.join(DOLTHUB_COLUMNS)} FROM option_chain "
                f"WHERE act_symbol = '{underlying.upper()}' AND date = '{day:%Y-%m-%d}' "
                f"ORDER BY expiration, strike, call_put LIMIT {limit} OFFSET {offset}", branch)
            collected.extend(rows)
            if len(rows) < limit:
                break
            offset += len(rows)
        if len(collected) > before_day:
            print(f"  {day:%Y-%m-%d}: {len(collected) - before_day} rows "
                  f"({len(collected)} total)", flush=True)
        day += timedelta(days=1)
    if not collected:
        return pl.DataFrame()
    return pl.DataFrame(collected, infer_schema_length=None)


def time_policy(args) -> ChainTimePolicy:
    hour, minute = (int(part) for part in args.quote_time.split(":"))
    return ChainTimePolicy(quote_time=time(hour, minute), expiry_time=time(hour, minute),
                           zone=args.zone)


def underlying_prices(path: Path | None) -> pl.DataFrame | None:
    """A `date,symbol,close` table, for chains that ship no underlying price."""
    if path is None:
        return None
    prices = pl.read_csv(path) if path.suffix == ".csv" else pl.read_parquet(path)
    column = pl.col("date")
    if prices.schema["date"] == pl.String:
        column = column.str.to_date(strict=False)
    return prices.with_columns(column.cast(pl.Date, strict=False))


def ingest(chain: pl.DataFrame, root: Path, *, prices: pl.DataFrame | None,
           keep_zero_bid: bool, structural_only: bool = False) -> None:
    """Clean one chunk and append it to the store.

    `structural_only` keeps everything the engine can represent — zero bids,
    wide spreads and IV outliers included — which is what a reusable store
    should hold: the liquidity filter belongs to the study, not to the archive,
    and a store built at one filter strength cannot be widened later.
    """
    if prices is not None:
        chain = with_underlying_prices(chain, prices)
    rules = (ChainFilter(drop_zero_bid=False, max_relative_spread=None, min_open_interest=0,
                         implied_volatility_bounds=None)
             if structural_only else ChainFilter(drop_zero_bid=not keep_zero_bid))
    cleaned, removed = clean_option_chain(chain, rules)
    if chain.height and not cleaned.height:
        # Silently writing nothing is the worst outcome here: DoltHub ships no
        # underlying price, so a forgotten --prices wipes out every row.
        hint = (" DoltHub chains carry no underlying price; pass --prices with a "
                "date,symbol,close table." if removed.get("missing_underlying") else "")
        raise RuntimeError(f"cleaning removed all {chain.height} rows ({removed}).{hint}")
    report = validate_option_chain(cleaned)
    for warning in report.warnings:
        print(f"  warning: {warning}")
    report.raise_if_failed()
    written = OptionChainStore(root).write(cleaned)
    print(f"  kept {cleaned.height} of {chain.height} rows "
          f"(removed: {removed or 'nothing'}) -> {len(written)} partition files under {root}")


def dolthub_prices(symbols: list[str], start: date, end: date, *, branch: str = DOLTHUB_BRANCH
                   ) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Unadjusted closes (`date,symbol,close`) and splits (`symbol,ex_date,factor`).

    The chain needs the traded price, so the closes are left unadjusted: a
    strike is compared with what the stock actually closed at that day. Splits
    come separately so a study can adjust *returns* without touching prices;
    `factor` is shares after per share before (4.0 for a 4-for-1). Closes are
    read one weekday per request for the reason `dolthub_rows` gives: `date` is
    the leading key, and a symbol-and-range query exceeds the server deadline.
    """
    for symbol in symbols:
        if not re.fullmatch(r"[A-Za-z0-9.\-]{1,10}", symbol):
            raise SystemExit(f"invalid symbol {symbol!r}")
    listed = ", ".join(f"'{symbol}'" for symbol in symbols)
    closes: list[dict[str, object]] = []
    day = start
    while day <= end:
        if day.weekday() < 5:
            closes += _dolthub_query(f"SELECT date, act_symbol AS symbol, close FROM ohlcv "
                                     f"WHERE date = '{day}' AND act_symbol IN ({listed})", branch,
                                     api=DOLTHUB_STOCKS_API)
        day += timedelta(days=1)
    splits = _dolthub_query(f"SELECT act_symbol AS symbol, ex_date, to_factor, for_factor FROM split "
                            f"WHERE act_symbol IN ({listed})", branch, api=DOLTHUB_STOCKS_API)
    prices = pl.DataFrame(closes, schema={"date": pl.Utf8, "symbol": pl.Utf8, "close": pl.Utf8}).with_columns(
        pl.col("date").str.to_date(), pl.col("close").cast(pl.Float64))
    factors = (pl.DataFrame(splits, schema={"symbol": pl.Utf8, "ex_date": pl.Utf8, "to_factor": pl.Utf8,
                                            "for_factor": pl.Utf8})
               .select("symbol", pl.col("ex_date").str.to_date(),
                       (pl.col("to_factor").cast(pl.Float64) / pl.col("for_factor").cast(pl.Float64)).alias("factor"))
               .filter(pl.col("ex_date").is_between(start, end)))
    return prices, factors


def main() -> None:
    # Shared as a parent so `--root` reads naturally on either side of the
    # subcommand; a store path typed after the file path is the common mistake.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--root", type=Path, default=Path("data/options"),
                        help="store root (default: data/options)")
    common.add_argument("--prices", type=Path,
                        help="date,symbol,close table used to fill underlying_price")
    common.add_argument("--keep-zero-bid", action="store_true",
                        help="keep zero-bid quotes instead of dropping them")
    common.add_argument("--structural-only", action="store_true",
                        help="drop only what the engine cannot represent, keeping zero bids, "
                             "wide spreads and IV outliers for the study to filter")
    common.add_argument("--zone", help="time zone for the vendor's time of day, "
                                       "e.g. America/New_York for a 16:00 ET close")
    common.add_argument("--quote-time", default="20:00",
                        help="time of day for a date-only quote column (default 20:00)")
    parser = argparse.ArgumentParser(description=__doc__, parents=[common],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    remote = sub.add_parser("dolthub", parents=[common],
                            help="query post-no-preference/options")
    remote.add_argument("--underlying", required=True)
    remote.add_argument("--start", required=True, type=date.fromisoformat)
    remote.add_argument("--end", required=True, type=date.fromisoformat)
    remote.add_argument("--branch", default=DOLTHUB_BRANCH)
    remote.add_argument("--page", type=int, default=DOLTHUB_ROW_CAP)
    remote.add_argument("--max-rows", type=int)

    stocks = sub.add_parser("dolthub-prices",
                            help="unadjusted closes and splits from post-no-preference/stocks")
    stocks.add_argument("--symbols", required=True, help="comma-separated tickers")
    stocks.add_argument("--start", required=True, type=date.fromisoformat)
    stocks.add_argument("--end", required=True, type=date.fromisoformat)
    stocks.add_argument("--output", type=Path, required=True,
                        help="directory for prices.csv and splits.csv")

    local = sub.add_parser("file", parents=[common],
                           help="normalise a downloaded chain file")
    local.add_argument("path", type=Path)
    local.add_argument("--source", default="kaggle_spy_eod",
                       choices=["kaggle_spy_eod", "dolthub", "long", "optionsdx"])
    local.add_argument("--underlying", help="required for a source with no ticker column")
    local.add_argument("--dates-per-chunk", type=int, default=20,
                       help="sessions normalised at a time; lower bounds memory further")

    args = parser.parse_args()
    if args.command == "dolthub-prices":
        closes, splits = dolthub_prices([s.strip().upper() for s in args.symbols.split(",")],
                                        args.start, args.end)
        args.output.mkdir(parents=True, exist_ok=True)
        closes.write_csv(args.output / "prices.csv")
        splits.write_csv(args.output / "splits.csv")
        print(f"{closes.height} closes, {splits.height} splits -> {args.output}")
        return
    prices = underlying_prices(args.prices)
    if args.command == "dolthub":
        print(f"{args.underlying} {args.start}..{args.end} from DoltHub", flush=True)
        raw = dolthub_rows(args.underlying, args.start, args.end, page=args.page,
                           max_rows=args.max_rows, branch=args.branch)
        if raw.is_empty():
            print("  no rows returned")
            return
        chain = normalize_option_chain(raw, "dolthub", time_policy=time_policy(args))
        ingest(chain, args.root, prices=prices, keep_zero_bid=args.keep_zero_bid,
               structural_only=args.structural_only)
    else:
        # Session-chunked by default: a Kaggle year does not have to fit in
        # memory to be ingested, and the store appends, so the result is the
        # same on disk either way.
        print(f"{args.path}", flush=True)
        for chunk in iter_normalized_chunks(args.path, args.source,
                                            dates_per_chunk=args.dates_per_chunk,
                                            underlying=args.underlying,
                                            time_policy=time_policy(args)):
            ingest(chunk, args.root, prices=prices, keep_zero_bid=args.keep_zero_bid,
                   structural_only=args.structural_only)
    print(f"done at {datetime.now(timezone.utc):%Y-%m-%d %H:%M:%SZ}")


if __name__ == "__main__":
    main()
