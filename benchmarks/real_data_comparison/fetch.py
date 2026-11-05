"""Download a real multi-symbol daily OHLCV universe and cache it to parquet.

Universe is the Nasdaq/NYSE symbol directory (plain pipe-delimited files, no
scraping), restricted to ordinary common shares, then whatever Yahoo actually
returns. Symbols that fail or are too short are dropped and counted, not faked.
"""
from __future__ import annotations
import io
import sys
import time
import urllib.request
from pathlib import Path
import pandas as pd
import yfinance as yf

HERE = Path(__file__).parent
OUT = HERE / "universe_daily.parquet"
START, END = "2015-01-01", "2026-09-01"
LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 1200


def _get(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as fh:
        return fh.read().decode("utf-8", "replace")


def universe() -> list[str]:
    syms: list[str] = []
    for name, col in (("nasdaqlisted", "Symbol"), ("otherlisted", "ACT Symbol")):
        txt = _get(f"https://www.nasdaqtrader.com/dynamic/symdir/{name}.txt")
        df = pd.read_csv(io.StringIO(txt), sep="|")
        df = df[df["Test Issue"] == "N"]
        if "ETF" in df.columns:
            df = df[df["ETF"] == "N"]
        col = col if col in df.columns else "Symbol"
        got = df[col].astype(str)
        got = got[~got.str.contains(r"[^A-Z]", regex=True)]  # drop units/warrants/pref
        syms += got.tolist()
    return sorted(set(syms))


def main() -> None:
    syms = universe()
    print(f"listed universe: {len(syms)} symbols; requesting first {LIMIT}", flush=True)
    syms = syms[:LIMIT]
    frames, failed, short = [], 0, 0
    t0 = time.time()
    for i in range(0, len(syms), 50):
        chunk = syms[i : i + 50]
        try:
            raw = yf.download(chunk, start=START, end=END, auto_adjust=False,
                              progress=False, threads=True, group_by="ticker")
        except Exception as exc:  # noqa: BLE001 - a dead ticker must not kill the run
            print(f"  chunk {i} failed: {exc}", flush=True)
            failed += len(chunk)
            continue
        level0 = set(raw.columns.get_level_values(0))
        for sym in chunk:
            if sym not in level0:
                failed += 1
                continue
            df = raw[sym].loc[:, ["Open", "High", "Low", "Close", "Volume"]].dropna()
            if len(df) < 500:
                short += 1
                continue
            df = df.reset_index()
            df.columns = ["ts", "open", "high", "low", "close", "volume"]
            df["symbol"] = sym
            frames.append(df)
        print(f"  {i+len(chunk)}/{len(syms)} kept={len(frames)} failed={failed} "
              f"short={short} {time.time()-t0:.0f}s", flush=True)
    out = pd.concat(frames, ignore_index=True)
    out = out[out[["open", "high", "low", "close"]].gt(0).all(axis=1)]
    out = out.sort_values(["ts", "symbol"], kind="stable").reset_index(drop=True)
    out.to_parquet(OUT, index=False)
    print(f"WROTE {OUT} rows={len(out):,} symbols={out.symbol.nunique()} "
          f"dropped_failed={failed} dropped_short={short} "
          f"span={out.ts.min().date()}..{out.ts.max().date()}")


main()
