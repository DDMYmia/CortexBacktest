"""Model-priced weekly option chains for an underlying with no stored chain history (TQQQ).

Quotes are Black-Scholes at the session close: IV = leverage x VXN x smile(z), the smile measured on
the real QQQ chain 2021-2022 (2-16 DTE) against VXN; z = ln(K/S) / (leverage x VXN x sqrt(T)).
Spread = clip(3% of theo, 0.01, 0.30); bid floors at 0 (a deep-OTM call can have no bid).
Weekly expiries (Friday, Thursday when Friday is not a session), out to 22 days. Raw (unsplit) prices,
so a window must not cross a split. BARS.csv rows are date,open,high,low,close without a header; VXN
(VXNCLS) and DTB3 are FRED CSVs. A research stand-in until real chains accumulate: the wheel's margin
stress on TQQQ runs on these, and the daily TWS snapshots are what will calibrate them.

    python -m cortexbacktest.options.model_chain BARS.csv VXN.csv DTB3.csv ROOT UNDERLYING LEVERAGE START END

The stores live in data/options-model/, one per split segment, listed with their windows and margin
leverage in data/options-model/index.json; the workspace's `scripts/strategies.py research` runs a
wheel spec on every segment of an underlying with no real chains.
"""
import os
import sys
from datetime import UTC, date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl
from scipy.stats import norm

from . import occ_symbol

NY = ZoneInfo("America/New_York")
SMILE_Z = np.array([-3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3])
SMILE_M = np.array([1.50, 1.36, 1.22, 1.11, 1.02, 0.94, 0.85, 0.775, 0.755, 0.83, 0.97, 1.14, 1.33])


def close(day: date) -> datetime:
    return datetime.combine(day, time(16), NY).astimezone(UTC)


def strikes(spot: float, right: str) -> np.ndarray:
    """Fixed strike increments, restricted to the current spot's moneyness window."""
    grid = np.concatenate([np.arange(0.5, 60, 0.5), np.arange(60, 200, 1.0), np.arange(200, 2000, 5.0)])
    low, high = (0.3, 2.5) if right == "PUT" else (0.3, 4.0)
    return grid[(grid >= spot * low) & (grid <= spot * high)]


def main(bars, vxn, dtb3, root, underlying, leverage, start, end):
    leverage = float(leverage)
    start, end = date.fromisoformat(start), date.fromisoformat(end)
    px = (pl.read_csv(bars, has_header=False, new_columns=["date", "open", "high", "low", "close"], try_parse_dates=True)
          .unique("date", keep="last").sort("date"))
    vol = pl.read_csv(vxn, try_parse_dates=True).rename({"value": "vxn"})
    rate = pl.read_csv(dtb3, try_parse_dates=True).rename({"value": "rate"})
    px = (px.join(vol, on="date", how="left").join(rate, on="date", how="left").sort("date")
          .with_columns(pl.col("vxn").forward_fill(), pl.col("rate").forward_fill())
          .filter(pl.col("date").is_between(start, end + timedelta(days=22))))
    sessions = px["date"].to_list()
    spot = dict(zip(sessions, px["close"].to_list()))
    sig = dict(zip(sessions, (px["vxn"] / 100).to_list()))
    r = dict(zip(sessions, (px["rate"] / 100).to_list()))
    weeks: dict[tuple, date] = {}
    for d in sessions:
        weeks[d.isocalendar()[:2]] = d  # last session of each week = its weekly expiry
    expiries = sorted(weeks.values())
    out = []
    for d in sessions:
        if d > end + timedelta(days=21):
            break
        S, v, rr = spot[d], sig[d] * leverage, r[d]
        quoted_at = close(d)
        for e in (e for e in expiries if 0 <= (e - d).days <= 22):
            expires_at = close(e)
            T = max((e - d).days, 0) / 365
            for right in ("PUT", "CALL"):
                K = strikes(S, right)
                if T > 0:
                    z = np.log(K / S) / (v * np.sqrt(T))
                    iv = v * np.interp(z, SMILE_Z, SMILE_M)
                    d1 = (np.log(S / K) + (rr + iv * iv / 2) * T) / (iv * np.sqrt(T))
                    d2 = d1 - iv * np.sqrt(T)
                    if right == "CALL":
                        theo, delta = S * norm.cdf(d1) - K * np.exp(-rr * T) * norm.cdf(d2), norm.cdf(d1)
                    else:
                        theo, delta = K * np.exp(-rr * T) * norm.cdf(-d2) - S * norm.cdf(-d1), norm.cdf(d1) - 1
                else:
                    iv = np.full_like(K, v)
                    theo = np.maximum(S - K, 0) if right == "CALL" else np.maximum(K - S, 0)
                    delta = np.where(theo > 0, 1.0 if right == "CALL" else -1.0, 0.0)
                half = np.clip(theo * 0.03, 0.01, 0.30) / 2
                bid = np.maximum(np.round(theo - half, 2), 0.0)
                bid = np.where(theo < 0.01, 0.0, bid)
                ask = np.maximum(np.round(theo + half, 2), 0.01)
                for k, b, a, i, dl in zip(K, bid, ask, iv, delta):
                    k = round(float(k), 2)
                    out.append((quoted_at, S, expires_at, (e - d).days, k, right, occ_symbol(underlying, expires_at, right, k),
                                float(b), float(a), (float(b) + float(a)) / 2, float(i), float(dl)))
    frame = pl.DataFrame(out, orient="row", schema=[
        "quote_date", "underlying_price", "expiry", "dte", "strike", "right", "option_symbol",
        "bid", "ask", "mid", "implied_volatility", "delta"]).with_columns(
        pl.col("dte").cast(pl.Int32),
        **{name: pl.lit(None, pl.Float64) for name in ("gamma", "theta", "vega", "rho", "volume", "open_interest")})
    for (year,), part in frame.group_by(pl.col("quote_date").dt.year()):
        folder = f"{root}/underlying={underlying}/year={year}"
        os.makedirs(folder, exist_ok=True)
        part.write_parquet(f"{folder}/part-synthetic.parquet")
    print(underlying, start, end, frame.height, "rows")  # noqa: T201


if __name__ == "__main__":
    main(*sys.argv[1:])
