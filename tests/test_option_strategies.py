from dataclasses import replace
from datetime import UTC, date, datetime, time, timedelta
from math import sin
from zoneinfo import ZoneInfo

import polars as pl
import pytest

from cortexbacktest.options import OptionChainStore, occ_symbol
from cortexbacktest.options.pricing import black_scholes, bsm_greeks
from cortexbacktest.options.strategies import (
    STRUCTURES, OptionCandidate, VolGate, leg_quotes, picks, plan, session_list, vol_features, window,
)

NEW_YORK = ZoneInfo("America/New_York")
FIRST, LAST = date(2024, 1, 2), date(2024, 6, 28)


def close(day: date) -> datetime:
    return datetime.combine(day, time(16), NEW_YORK).astimezone(UTC)


def synthetic_chain(underlying: str, base: float) -> pl.DataFrame:
    """Weekday sessions, Friday expiries out to 60 days, Black-Scholes quotes around a wavy spot."""
    sessions = [FIRST + timedelta(days=i) for i in range((LAST - FIRST).days + 1)]
    sessions = [d for d in sessions if d.weekday() < 5]
    rows = []
    for n, day in enumerate(sessions):
        spot = base * (1 + 0.08 * sin(n / 9) + 0.0004 * n)
        vol = 0.18 + 0.06 * sin(n / 23)
        for expiry in (d for d in sessions if d.weekday() == 4 and 0 < (d - day).days <= 60):
            years = (expiry - day).days / 365
            for step in range(-12, 13):
                strike = round(base * (1 + step * 0.025), 2)
                for right in ("CALL", "PUT"):
                    price = black_scholes(right, spot, strike, years, 0.0, vol)
                    greeks = bsm_greeks(right, spot, strike, years, 0.0, vol)
                    bid, ask = max(price * 0.97, 0.0), max(price * 1.03, 0.01)
                    rows.append({
                        "quote_date": close(day), "underlying": underlying, "underlying_price": spot,
                        "expiry": close(expiry), "dte": (expiry - day).days, "strike": strike, "right": right,
                        "option_symbol": occ_symbol(underlying, expiry, right, strike),
                        "bid": bid, "ask": ask, "mid": (bid + ask) / 2, "implied_volatility": vol,
                        "delta": greeks.delta, "gamma": greeks.gamma, "theta": greeks.theta,
                        "vega": greeks.vega, "rho": greeks.rho, "volume": 100.0, "open_interest": 1000.0})
    return pl.DataFrame(rows).with_columns(pl.col("dte").cast(pl.Int32))


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    root = tmp_path_factory.mktemp("chains")
    store = OptionChainStore(root)
    for underlying, base in (("SPY", 100.0), ("QQQ", 60.0)):
        frame = synthetic_chain(underlying, base)
        for (year,), part in frame.group_by(pl.col("quote_date").dt.year()):
            folder = root / f"underlying={underlying}" / f"year={year}"
            folder.mkdir(parents=True, exist_ok=True)
            part.drop("underlying").write_parquet(folder / "part-test.parquet")
    return store



def test_candidate_rejects_wings_inside_the_short_strike():
    with pytest.raises(ValueError):
        OptionCandidate("iron_condor", 30, delta=0.10, wing=0.20)


def test_picks_place_each_leg_by_delta_at_one_expiry_with_wings_further_out(store):
    start, end = window(FIRST, LAST)
    condor = OptionCandidate("iron_condor", 30, delta=0.25, wing=0.08)
    chosen = picks(store, condor, "SPY", start=start, end=end)
    assert chosen.height > 50
    legs = STRUCTURES["iron_condor"]
    put_short, put_wing, call_short, call_wing = (legs.index(leg) for leg in legs)
    assert (chosen[f"strike{put_wing}"] < chosen[f"strike{put_short}"]).all()
    assert (chosen[f"strike{call_wing}"] > chosen[f"strike{call_short}"]).all()
    assert (chosen[f"strike{put_short}"] < chosen[f"strike{call_short}"]).all()


def test_plan_never_overlaps_and_profit_take_closes_before_expiry(store):
    start, end = window(FIRST, LAST)
    sessions = session_list(store, "SPY", start=start, end=end)
    features = vol_features(store, "SPY", start=start, end=end)
    candidate = OptionCandidate("short_put", 30, delta=0.30, profit_take=0.5)
    chosen = picks(store, candidate, "SPY", start=start, end=end)
    symbols = sorted(set(chosen["symbol0"].to_list()))
    quotes = leg_quotes(store, "SPY", symbols, start=start, end=end + timedelta(days=60))
    positions, _ = plan(candidate, chosen, features, quotes, sessions)
    assert len(positions) >= 3
    for earlier, later in zip(positions, positions[1:]):
        assert later.entered > (earlier.exit_at or earlier.expiry)
    taken = [p for p in positions if p.exit_at is not None]
    assert taken and all(p.entered < p.exit_at < p.expiry for p in taken)
    held, _ = plan(replace(candidate, profit_take=None), chosen, features, quotes, sessions)
    assert all(p.exit_at is None for p in held) and len(held) < len(positions)


def test_a_gate_that_never_opens_takes_no_position(store):
    start, end = window(FIRST, LAST)
    sessions = session_list(store, "SPY", start=start, end=end)
    features = vol_features(store, "SPY", start=start, end=end)
    candidate = OptionCandidate("short_put", 30, gate=VolGate("iv_hv", "above", 1e9))
    chosen = picks(store, candidate, "SPY", start=start, end=end)
    positions, dropped = plan(candidate, chosen, features, {}, sessions)
    assert positions == [] and dropped["gated"] > 0


def test_features_are_computed_from_the_session_and_its_past(store):
    start, end = window(date(2024, 3, 1), LAST)
    features = vol_features(store, "SPY", start=start, end=end, hv_window=10, rank_window=20)
    assert features["quote_date"].min() >= start
    assert features["atm_iv"].drop_nulls().is_between(0.1, 0.3).all()
    assert features["iv_rank"].drop_nulls().is_between(0.0, 1.0).all()
    # A later session must not change an earlier one's value.
    shorter = vol_features(store, "SPY", start=start, end=end - timedelta(days=30), hv_window=10, rank_window=20)
    joined = shorter.join(features, on="quote_date", suffix="_full")
    assert (joined["iv_rank"] - joined["iv_rank_full"]).abs().max() == 0



def test_splits_adjust_returns_but_never_the_traded_price(store):
    start, end = window(FIRST, LAST)
    ex_date = close(date(2024, 4, 1))
    plain = vol_features(store, "SPY", start=start, end=end)
    split = vol_features(store, "SPY", start=start, end=end, splits=[(ex_date, 2.0)])
    assert split["spot"].equals(plain["spot"])
    before = split.filter(pl.col("quote_date") < ex_date)
    after = split.filter(pl.col("quote_date") >= ex_date)
    assert (before["adjusted"] * 2 - before["spot"]).abs().max() < 1e-9
    assert (after["adjusted"] - after["spot"]).abs().max() == 0


def test_volatility_from_a_three_day_week_matches_the_daily_series():
    import numpy as np

    from cortexbacktest.options.strategies import _trading_day_windows

    rng = np.random.default_rng(3)
    days = [close(d) for d in (date(2021, 1, 4) + timedelta(days=i) for i in range(900)) if d.weekday() < 5]
    closes = 100 * np.exp(np.cumsum(rng.normal(0, 0.2 / np.sqrt(252), len(days))))
    iv = np.full(len(days), 0.2)
    daily, _ = _trading_day_windows(days, closes, iv, hv_window=63, rank_window=252)
    keep = [i for i, day in enumerate(days) if day.weekday() in (0, 2, 4)]
    sparse, _ = _trading_day_windows([days[i] for i in keep], closes[keep], iv[keep], hv_window=63, rank_window=252)
    assert abs(np.nanmedian(daily) - 0.2) < 0.02
    assert abs(np.nanmedian(sparse) - np.nanmedian(daily)) < 0.02


def test_a_days_to_expiry_exit_closes_before_the_threshold_and_before_expiry(store):
    start, end = window(FIRST, LAST)
    sessions = session_list(store, "SPY", start=start, end=end)
    features = vol_features(store, "SPY", start=start, end=end)
    candidate = OptionCandidate("short_put", 30, delta=0.30, exit_dte=14)
    chosen = picks(store, candidate, "SPY", start=start, end=end)
    quotes = leg_quotes(store, "SPY", sorted(set(chosen["symbol0"].to_list())), start=start,
                        end=end + timedelta(days=60))
    positions, _ = plan(candidate, chosen, features, quotes, sessions)
    assert positions and all(p.exit_at is not None for p in positions)
    for p in positions:
        assert (p.expiry - p.exit_at).days <= 14
        earlier = [s for s in sessions if p.entered < s < p.exit_at]
        assert all((p.expiry - s).days > 14 for s in earlier)
