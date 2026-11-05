"""Shared scenario definition for the real-data engine comparison.

The shared signal scenario is described below; fill conventions differ:

  * Real daily OHLCV inputs; runner-specific calendar alignment may fill gaps.
  * SMA(20) vs SMA(50) on close, long-only, 100 shares per symbol.
  * The signal is read on bar T's close and the resulting order fills at bar
    T+1's open. Engines that cannot express next-bar-open fills say so in
    their own note rather than silently using a different convention.
  * Zero commission, zero slippage, cash large enough that nothing is
    rejected for funding.
  * Data construction (DataFrames, feeds, bundles) happens OUTSIDE the timer.
    Only the run itself is timed.

The driver reads the last JSON object from stdout and compares order counts
within a tolerance; this does not establish fill or accounting equivalence.
"""

from __future__ import annotations

import argparse
import math
import json
import sys
from pathlib import Path

import numpy as np

FAST, SLOW, QTY = 20, 50, 100
INITIAL_CASH = 1_000_000_000.0
HERE = Path(__file__).resolve().parent


def load(dataset: str, symbols: int, bars: int):
    """Return (symbol_names, ts_int64_ns, dict_of_2d_arrays) sliced to shape."""
    raw = np.load(HERE / dataset, allow_pickle=False)
    names = [str(s) for s in raw["symbols"][:symbols]]
    if len(names) < symbols:
        raise SystemExit(f"dataset has {len(names)} symbols, need {symbols}")
    ts = raw["ts"][-bars:]
    cols = {c: raw[c][:symbols, -bars:] for c in ("open", "high", "low", "close", "volume")}
    return names, ts, cols


def frames(dataset: str, symbols: int, bars: int):
    """Per-symbol pandas frames with a DatetimeIndex — the common input shape."""
    import pandas as pd

    names, ts, cols = load(dataset, symbols, bars)
    index = pd.DatetimeIndex(ts.astype("datetime64[ns]"), name="Date")
    out = {}
    for i, sym in enumerate(names):
        out[sym] = pd.DataFrame(
            {
                "Open": cols["open"][i],
                "High": cols["high"][i],
                "Low": cols["low"][i],
                "Close": cols["close"][i],
                "Volume": cols["volume"][i],
            },
            index=index,
        )
    return out


def signals(close: np.ndarray) -> np.ndarray:
    """Boolean desired-long matrix, same arithmetic for every engine.

    Shape matches `close` (symbols x bars). True means "want to be long at the
    open of the NEXT bar". Warm-up bars are False.
    """
    close = np.atleast_2d(close)
    fast = _sma(close, FAST)
    slow = _sma(close, SLOW)
    want = fast > slow
    want[:, : SLOW - 1] = False
    return want


def _sma(a: np.ndarray, n: int) -> np.ndarray:
    csum = np.cumsum(a, axis=1)
    out = np.full_like(a, np.nan)
    out[:, n - 1] = csum[:, n - 1] / n
    out[:, n:] = (csum[:, n:] - csum[:, :-n]) / n
    return out


def expected_orders(want: np.ndarray) -> int:
    """How many orders the scenario must produce, from the signal alone.

    Every runner is checked against this. An engine that aborts early — as
    nautilus does when its cash account goes negative — otherwise looks
    spectacularly fast while having replayed only part of the data.
    """
    prev = np.hstack([np.zeros((want.shape[0], 1), bool), want[:, :-1]])
    return int((want != prev).sum())


# Fill conventions. A decision is made on bar T after its close.
NEXT_OPEN, NEXT_CLOSE, SAME_CLOSE = "next-open", "next-close", "same-bar-close"
CONVENTIONS = (NEXT_OPEN, NEXT_CLOSE, SAME_CLOSE)
# Resting scenario: every bar, every symbol submits a GTC buy limit at this
# fraction of the bar's close. Nothing is cancelled; fills accumulate.
REST_FRACTION = 0.97


def cli() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="wide.npz")
    p.add_argument("--symbols", type=int, required=True)
    p.add_argument("--bars", type=int, required=True)
    p.add_argument("--scenario", choices=("sma", "resting", "mtf"), default="sma")
    p.add_argument("--convention", choices=CONVENTIONS, default=NEXT_OPEN,
                   help="Only runners that can switch convention read this.")
    return p.parse_args()


class Mtf:
    """Multi-timeframe strategy, computed bar by bar inside every engine's callback.

    Weekly regime: the last completed week's close above its 10-week average.
    Daily: Wilder RSI(14), Bollinger(20, 2) and Wilder ATR(14). Enter when the
    regime is up and RSI < 35 or the close is under the lower band; exit when
    RSI > 65, the close is over the 20-day mean, or the regime turns down. The
    position is floor(RISK / ATR) shares, capped at floor(CAP / close), so every
    trade has its own size. `update` returns the target position in shares.
    """

    RSI_N, BAND_N, ATR_N, WEEKS = 14, 20, 14, 10
    RISK, CAP = 200.0, 20_000.0

    def __init__(self, week_ids):
        self.week_ids = week_ids
        self.state = {}

    def update(self, symbol, i, o, h, low, c) -> int:
        st = self.state.get(symbol)
        if st is None:
            st = self.state[symbol] = {"closes": [], "weekly": [], "week": None, "week_close": None,
                                       "gain": None, "loss": None, "atr": None, "prev": None, "target": 0}
        week = self.week_ids[i]
        if st["week"] is not None and week != st["week"]:
            st["weekly"].append(st["week_close"])
        st["week"], st["week_close"] = week, c
        prev = st["prev"]
        if prev is not None:
            change = c - prev
            gain, loss = max(change, 0.0), max(-change, 0.0)
            true_range = max(h - low, abs(h - prev), abs(low - prev))
            if st["gain"] is None:
                st["gain"], st["loss"], st["atr"] = gain, loss, true_range
            else:
                n, m = self.RSI_N, self.ATR_N
                st["gain"] = (st["gain"] * (n - 1) + gain) / n
                st["loss"] = (st["loss"] * (n - 1) + loss) / n
                st["atr"] = (st["atr"] * (m - 1) + true_range) / m
        st["prev"] = c
        closes = st["closes"]
        closes.append(c)
        if len(closes) > self.BAND_N:
            del closes[0]
        weekly = st["weekly"]
        if len(closes) < self.BAND_N or len(weekly) < self.WEEKS or st["atr"] is None:
            return st["target"]
        # math.fsum is correctly rounded on every Python; the built-in sum() of floats
        # changed in 3.12, and engines here run on 3.11 and 3.13 interpreters.
        mean = math.fsum(closes) / self.BAND_N
        std = (math.fsum((x - mean) ** 2 for x in closes) / self.BAND_N) ** 0.5
        rsi = 100.0 if st["loss"] == 0 else 100.0 - 100.0 / (1.0 + st["gain"] / st["loss"])
        regime = weekly[-1] > math.fsum(weekly[-self.WEEKS:]) / self.WEEKS
        if st["target"] == 0 and regime and (rsi < 35 or c < mean - 2 * std):
            st["target"] = int(min(self.RISK / max(st["atr"], 1e-9), self.CAP / c))
        elif st["target"] > 0 and (rsi > 65 or c > mean or not regime):
            st["target"] = 0
        return st["target"]


def week_ids(ts) -> list[int]:
    """ISO year*100 + week for each bar timestamp (ns since epoch)."""
    import datetime as _dt
    out = []
    for t in ts:
        iso = _dt.datetime.fromtimestamp(int(t) // 1_000_000_000, _dt.UTC).isocalendar()
        out.append(iso[0] * 100 + iso[1])
    return out


def mtf_targets(ts, cols) -> np.ndarray:
    """The Mtf target positions for every symbol and bar, run outside any engine."""
    weeks = week_ids(ts)
    model = Mtf(weeks)
    o, h, low, c = (cols[k] for k in ("open", "high", "low", "close"))
    out = np.zeros(c.shape, dtype=np.int64)
    for s in range(c.shape[0]):
        for i in range(c.shape[1]):
            out[s, i] = model.update(s, i, float(o[s, i]), float(h[s, i]), float(low[s, i]), float(c[s, i]))
    return out


class MtfFeed:
    """What a runner calls from its per-bar callback: `target(symbol_index, bar_index)`.

    Prices come from the shared arrays rather than each engine's bar object, so
    every engine computes on identical inputs (nautilus rounds its bar prices).
    """

    def __init__(self, ts, cols):
        self.model = Mtf(week_ids(ts))
        self.o, self.h, self.low, self.c = (cols[k].tolist() for k in ("open", "high", "low", "close"))

    def target(self, s: int, i: int) -> int:
        return self.model.update(s, i, self.o[s][i], self.h[s][i], self.low[s][i], self.c[s][i])


def expected_trades(targets: np.ndarray) -> int:
    """Orders the Mtf scenario must produce: one per change in target position."""
    prev = np.hstack([np.zeros((targets.shape[0], 1), targets.dtype), targets[:, :-1]])
    return int((targets != prev).sum())


def expected_for(scenario: str, ts, cols, want) -> int | None:
    """Order count a runner must report, or None where the scenario has no fixed count."""
    if scenario == "sma":
        return expected_orders(np.atleast_2d(want))
    if scenario == "mtf":
        return expected_trades(mtf_targets(ts, cols))
    return None


def truth_pnl(cols: dict, want: np.ndarray, convention: str) -> float:
    """P&L of the SMA scenario from a plain ledger, per fill convention.

    Holding on bar t is the decision of bar t-1 (next-bar conventions) or of
    bar t itself (same-bar close). A decision on the final bar of a next-bar
    convention has no bar left to fill on, so it is not traded. The final
    position is marked at the last close.
    """
    close, opn = cols["close"], cols["open"]
    # Boolean signals are 100-share positions; integer arrays are share targets.
    shares = want.astype(float) * QTY if want.dtype == bool else want.astype(float)
    if convention == SAME_CLOSE:
        held, price = shares, close
    else:
        held = np.hstack([np.zeros((shares.shape[0], 1)), shares[:, :-1]])
        price = opn if convention == NEXT_OPEN else close
    traded = np.diff(held, axis=1, prepend=0.0)
    return float(-(traded * price).sum() + (held[:, -1] * close[:, -1]).sum())


def unsupported(engine: str, reason: str) -> None:
    """Report that an engine cannot express the scenario, and stop."""
    json.dump({"engine": engine, "unsupported": reason}, sys.stdout)
    sys.stdout.write("\n")
    raise SystemExit(0)


def emit(engine: str, seconds: float, orders: int, symbols: int, bars: int, note: str = "",
         expected: int | None = None, *, pnl: float | None = None, convention: str = "",
         fills: int | None = None) -> None:
    """Print one result row. `pnl` is final equity minus starting cash."""
    json.dump(
        {
            "engine": engine,
            "pnl": None if pnl is None else round(float(pnl), 4),
            "convention": convention,
            "fills": fills,
            "expected_orders": expected,
            "seconds": round(seconds, 6),
            "orders": int(orders),
            "symbols": symbols,
            "bars_per_symbol": bars,
            "bar_observations": symbols * bars,
            "observations_per_second": round(symbols * bars / seconds) if seconds > 0 else None,
            "note": note,
        },
        sys.stdout,
    )
    sys.stdout.write("\n")
