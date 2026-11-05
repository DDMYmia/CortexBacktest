"""One session of a stored option chain, shaped the way option-chain pages show it.

`chain_view` reads the `OptionChainStore` for one underlying and returns plain JSON:

- `expiries`: every expiry quoted that session with its days to expiry, contract count
  and at-the-money implied volatility -- the term structure;
- `rows`: one row per strike of the selected expiry, the call on the left and the put on
  the right (bid, ask, mid, IV, Greeks, volume, open interest, probability in the money);
- `summary`: spot, at-the-money IV, the expected move (the at-the-money straddle's mid),
  put/call open interest and volume ratios, and the IV rank and percentile of the 30-day
  at-the-money IV against the stored lookback, including the selected session.

At the money is the listed strike nearest the underlying price; its IV is the mean of the
call's and the put's where both are quoted. The probability in the money is the
risk-neutral N(d2) from the contract's own IV with rates and dividends set to zero, so it
is a market-implied odds, not a forecast. The session is the last one quoted on or before
the requested date; without an expiry, the first one at least 20 days out is shown.
"""
from datetime import date, datetime, time, timedelta, timezone
from math import erf, log, sqrt
from pathlib import Path
from typing import Any

import polars as pl

from .chains import OptionChainStore

DEFAULT_ROOT = Path(__file__).resolve().parents[3] / "data" / "options"
STRIKES = 41  # strikes shown, centred on the money
HISTORY_DAYS = 365
TARGET_DTE, DTE_BAND = 30, (20, 45)
QUOTE_COLUMNS = ("bid", "ask", "mid", "implied_volatility", "delta", "gamma", "theta", "vega",
                 "volume", "open_interest")


def _end_of(day: date) -> datetime:
    return datetime.combine(day, time.max, timezone.utc)


def _prob_itm(right: str, spot: float, strike: float, years: float, vol: float | None) -> float | None:
    if vol is None or vol <= 0 or years <= 0:
        return None
    d2 = (log(spot / strike) - vol * vol * years / 2) / (vol * sqrt(years))
    return 0.5 * (1 + erf((d2 if right == "CALL" else -d2) / sqrt(2)))


def _atm_iv(quotes: pl.DataFrame) -> pl.DataFrame:
    """At-the-money IV per (quote_date, expiry): call/put mean at the strike nearest spot."""
    return (quotes.with_columns((pl.col("strike") - pl.col("underlying_price")).abs().alias("_gap"))
            .filter(pl.col("_gap") == pl.col("_gap").min().over("quote_date", "expiry"))
            .filter(pl.col("strike") == pl.col("strike").min().over("quote_date", "expiry"))
            .filter(pl.col("implied_volatility") >= 0)
            .group_by("quote_date", "expiry")
            .agg(pl.col("implied_volatility").mean().alias("atm_iv"), pl.col("strike").first().alias("atm_strike")))


def _iv_history(history: pl.DataFrame) -> pl.DataFrame:
    """The 30-day at-the-money IV of each session: the expiry nearest 30 days inside 20-45."""
    near = history.filter(pl.col("dte").is_between(*DTE_BAND))
    atm = _atm_iv(near).join(near.select("quote_date", "expiry", "dte").unique(), on=["quote_date", "expiry"])
    return (atm.with_columns((pl.col("dte") - TARGET_DTE).abs().alias("_off"))
            .filter(pl.col("_off") == pl.col("_off").min().over("quote_date"))
            .sort("expiry").group_by("quote_date").agg(pl.col("atm_iv").first()).sort("quote_date"))


def coverage(store: OptionChainStore, underlying: str) -> tuple[date, date] | None:
    years = sorted(int(p.name.split("=")[1]) for p in (store.root / f"underlying={underlying}").glob("year=*"))
    if not years:
        return None
    first = store.scan(underlyings=[underlying], end=_end_of(date(years[0], 12, 31))).select(
        pl.col("quote_date").min()).collect().item()
    last = store.scan(underlyings=[underlying], start=datetime(years[-1], 1, 1, tzinfo=timezone.utc)).select(
        pl.col("quote_date").max()).collect().item()
    return first.date(), last.date()


def chain_view(request: dict[str, Any]) -> dict[str, Any]:
    underlying = str(request["underlying"]).upper()
    store = OptionChainStore(request.get("root") or DEFAULT_ROOT)
    span = coverage(store, underlying)
    if span is None:
        return {"underlying": underlying, "coverage": None}
    wanted = date.fromisoformat(request["date"]) if request.get("date") else span[1]
    if wanted < span[0]:
        raise ValueError("option_chain_session_unavailable")
    wanted = min(wanted, span[1])
    history = store.scan(underlyings=[underlying], start=_end_of(wanted - timedelta(days=HISTORY_DAYS + 7)),
                         end=_end_of(wanted)).collect()
    if history.is_empty():
        raise ValueError("option_chain_session_unavailable")
    session = history.select(pl.col("quote_date").max()).item()
    day = history.filter(pl.col("quote_date") == session)
    spot = float(day.select(pl.col("underlying_price").median()).item())

    atm = _atm_iv(day)
    expiries = (day.group_by("expiry").agg(pl.col("dte").first(), pl.len().alias("contracts"))
                .join(atm, on="expiry", how="left").sort("expiry"))
    chosen = request.get("expiry")
    if chosen:
        pick = expiries.filter(pl.col("expiry").dt.date() == date.fromisoformat(chosen))
        if pick.is_empty():
            raise ValueError("option_chain_expiry_unavailable")
    else:
        pick = expiries.filter(pl.col("dte") >= DTE_BAND[0]).head(1)
    if pick.is_empty():
        pick = expiries.head(1)
    expiry, dte = pick["expiry"][0], int(pick["dte"][0])
    years = dte / 365

    chain = day.filter(pl.col("expiry") == expiry)
    strikes = sorted(set(chain["strike"].to_list()), key=lambda k: (abs(k - spot), k))[:STRIKES]
    sides = {right: {row["strike"]: row for row in chain.filter(pl.col("right") == right).iter_rows(named=True)}
             for right in ("CALL", "PUT")}
    rows = []
    for strike in sorted(strikes):
        row: dict[str, Any] = {"strike": strike}
        for right, prefix in (("CALL", "call"), ("PUT", "put")):
            quote = sides[right].get(strike)
            if quote is None:
                continue
            row.update({f"{prefix}_{name}": quote[name] for name in QUOTE_COLUMNS})
            row[f"{prefix}_prob_itm"] = _prob_itm(right, spot, strike, years, quote["implied_volatility"])
        rows.append(row)

    atm_strike = min(strikes, key=lambda k: abs(k - spot))
    straddle = [sides[r].get(atm_strike, {}).get("mid") for r in ("CALL", "PUT")]
    iv_series = _iv_history(history)
    current = iv_series.filter(pl.col("quote_date") == session)["atm_iv"]
    rank = percentile = None
    if not current.is_empty() and iv_series.height > 1:
        now, past = float(current[0]), iv_series["atm_iv"]
        low, high = iv_series.select(pl.col("atm_iv").min().alias("low"), pl.col("atm_iv").max().alias("high")).row(0)
        rank = (now - low) / (high - low) if high > low else None
        percentile = float((past < now).cast(pl.Float64).sum()) / past.len()
    totals = chain.group_by("right").agg(
        pl.when(pl.col(field).is_not_null().all()).then(pl.col(field).sum()).otherwise(None).alias(field)
        for field in ("open_interest", "volume"))
    total = {r["right"]: r for r in totals.iter_rows(named=True)}

    def ratio(field: str) -> float | None:
        call, put = total.get("CALL", {}).get(field), total.get("PUT", {}).get(field)
        return put / call if call and put is not None else None

    atm_row = atm.filter(pl.col("expiry") == expiry)
    return {
        "underlying": underlying, "coverage": [span[0].isoformat(), span[1].isoformat()],
        "session": session.date().isoformat(), "expiry": expiry.date().isoformat(), "dte": dte,
        "summary": {
            "spot": spot, "atm_strike": atm_strike,
            "atm_iv": float(atm_row["atm_iv"][0]) if not atm_row.is_empty() else None,
            "expected_move": straddle[0] + straddle[1] if straddle[0] is not None and straddle[1] is not None else None,
            "iv30": float(current[0]) if not current.is_empty() else None, "iv_rank": rank, "iv_percentile": percentile,
            "put_call_oi": ratio("open_interest"), "put_call_volume": ratio("volume"),
        },
        "expiries": [{"expiry": r["expiry"].date().isoformat(), "dte": r["dte"], "contracts": r["contracts"],
                      "atm_iv": r["atm_iv"]} for r in expiries.iter_rows(named=True)],
        "rows": rows,
        "iv_history": [{"date": r["quote_date"].date().isoformat(), "atm_iv": r["atm_iv"]}
                       for r in iv_series.iter_rows(named=True)],
    }
