"""Flatten the real universes into aligned per-symbol arrays.

From `universe_daily.parquet` (fetch.py, Yahoo daily 2015-2026), two datasets,
because requiring a common calendar trades symbol count against history depth:
  deep.npz  - fewer symbols, full history
  wide.npz  - as many symbols as possible over a shorter common window
From `recent_daily.parquet` (export_daily.py, the 1-minute store, 2024-01..2026-03):
  recent.npz - every full-calendar ticker without a split-like move
Symbols missing any bar on the chosen calendar are dropped, not forward-filled.
"""
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
# The minute store is unadjusted; a >40% one-day move is treated as a split and
# the ticker dropped, so no engine trades through one.
SPLIT_MOVE = 0.40


def write(name: str, frame: pd.DataFrame, dates: np.ndarray) -> None:
    keep = sorted(frame.symbol.unique())
    frame = frame.sort_values(["symbol", "ts"], kind="stable")
    cols = {c: frame[c].to_numpy(np.float64).reshape(len(keep), len(dates))
            for c in ("open", "high", "low", "close", "volume")}
    np.savez(HERE / name, symbols=np.array(keep), ts=dates.astype("datetime64[ns]").astype("int64"), **cols)
    print(f"WROTE {name}  {len(keep)} symbols x {len(dates)} bars = {len(keep)*len(dates):,} real bar observations")


def build(df: pd.DataFrame, name: str, n_bars: int) -> None:
    dates = np.sort(df.ts.unique())[-n_bars:]
    sub = df[df.ts.isin(dates)]
    counts = sub.groupby("symbol").size()
    write(name, sub[sub.symbol.isin(counts[counts == n_bars].index)], dates)


def build_recent() -> None:
    df = pd.read_parquet(HERE / "recent_daily.parquet").rename(columns={"ticker": "symbol", "day": "ts"})
    df["ts"] = pd.to_datetime(df["ts"])
    dates = np.sort(df.ts.unique())
    counts = df.groupby("symbol").size()
    df = df[df.symbol.isin(counts[counts == len(dates)].index)].sort_values(["symbol", "ts"])
    move = df.groupby("symbol")["close"].pct_change().abs()
    split_like = set(df.loc[move > SPLIT_MOVE, "symbol"])
    print(f"recent: dropped split-like tickers {sorted(split_like)}")
    write("recent.npz", df[~df.symbol.isin(split_like)], dates)


if (HERE / "universe_daily.parquet").exists():
    universe = pd.read_parquet(HERE / "universe_daily.parquet")
    build(universe, "deep.npz", 2932)
    build(universe, "wide.npz", 1000)
if (HERE / "recent_daily.parquet").exists():
    build_recent()
