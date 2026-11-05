"""Risk review of a live stock-and-option book: exposures, expiry, cover, scenarios, margin.

`analyse` turns one collected account (positions, the broker's Greeks and IVs, account
values, daily closes) plus an event list into a report:

- exposures per underlying: shares, option delta, dollar delta, the dollar delta in
  benchmark terms (times the name's beta), gamma per 1% move, vega, theta;
- the front expiry when it settles today: which short options finish in the money at the
  current price (certain) or within `borderline` of it (possible), with the cash each
  assignment moves;
- covered-call cover at the next expiry for the shares held after settlement;
- scenarios: the benchmark moves by each shock and every name by beta times it (floored at
  -95%); profit and loss at the next expiry and mid-way, and the Reg-T maintenance
  requirement and excess liquidity right after an instant shock (IV raised on drops);
- the benchmark move at which excess liquidity reaches zero (forced liquidation);
- reference tables: index puts sized against the benchmark exposure, and covered calls for
  the uncovered shares.

The margin model is `RegTOptionMargin` with one free parameter, the factor applied to the
leveraged ETFs, fitted each run so the model reproduces the broker's maintenance
requirement for the book as it stands; the fit and its residual are reported. Options are
repriced with Black-Scholes at each contract's own IV. This is a research estimate, not
the broker's risk engine and not advice.
"""
from __future__ import annotations

import statistics
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from math import ceil, erf, log, sqrt
from typing import Any
from zoneinfo import ZoneInfo

from .contracts import OptionContract
from .margin import RegTOptionMargin
from .pricing import OptionRight, black_scholes, bsm_greeks

NEW_YORK = ZoneInfo("America/New_York")
YEAR = 365 * 86400


@dataclass(frozen=True, slots=True)
class Leg:
    symbol: str
    quantity: float
    expiry: date | None = None  # None: stock
    strike: float = 0.0
    right: str | None = None    # "CALL" / "PUT"
    iv: float | None = None
    delta: float | None = None
    gamma: float | None = None
    vega: float | None = None
    theta: float | None = None
    average_cost: float | None = None

    @property
    def option(self) -> bool:
        return self.expiry is not None


def close_of(day: date) -> datetime:
    return datetime.combine(day, time(16), NEW_YORK).astimezone(timezone.utc)


def years_to(day: date, now: datetime) -> float:
    return max((close_of(day) - now).total_seconds() / YEAR, 1e-6)


def betas(history: dict[str, list[list[Any]]], benchmark: str) -> dict[str, dict[str, float]]:
    """Beta to the benchmark and annualised volatility from daily closes, last 252 returns."""
    def returns(rows: list[list[Any]]) -> dict[str, float]:
        rows = sorted(rows)[-253:]
        return {rows[i][0]: rows[i][1] / rows[i - 1][1] - 1 for i in range(1, len(rows)) if rows[i - 1][1]}
    base = returns(history[benchmark])
    out = {}
    for symbol, rows in history.items():
        r = returns(rows)
        common = [d for d in r if d in base]
        if len(common) < 60:
            continue
        x, y = [base[d] for d in common], [r[d] for d in common]
        mx, my = statistics.fmean(x), statistics.fmean(y)
        out[symbol] = {"beta": sum((a - mx) * (b - my) for a, b in zip(x, y)) / sum((a - mx) ** 2 for a in x),
                       "vol": statistics.pstdev(y) * sqrt(252), "sessions": len(common)}
    return out


def _price(leg: Leg, spot: float, now: datetime, rate: float, iv_multiplier: float = 1.0) -> float:
    assert leg.expiry is not None and leg.right is not None
    if leg.iv is None:
        raise ValueError("portfolio_option_iv_unavailable")
    return black_scholes(OptionRight(leg.right), spot, leg.strike, years_to(leg.expiry, now), rate,
                         leg.iv * iv_multiplier)


def exposures(legs: list[Leg], spots: dict[str, float], beta: dict[str, float]) -> dict[str, Any]:
    rows: dict[str, defaultdict[str, Any]] = {}
    for leg in legs:
        row = rows.setdefault(leg.symbol, defaultdict(float))
        if not leg.option:
            row["shares"] += leg.quantity
            continue
        size, spot = leg.quantity * 100, spots[leg.symbol]
        for field, amount in (
            ("option_delta", None if leg.delta is None else size * leg.delta),
            ("gamma_per_1pct", None if leg.gamma is None else size * leg.gamma * spot * spot * 0.01),
            ("vega", None if leg.vega is None else size * leg.vega),
            ("theta", None if leg.theta is None else size * leg.theta),
        ):
            row[field] = None if row[field] is None or amount is None else row[field] + amount
    table = []
    for symbol, row in sorted(rows.items()):
        net = None if row["option_delta"] is None else row["shares"] + row["option_delta"]
        dollars = None if net is None else net * spots[symbol]
        table.append({"symbol": symbol, "spot": spots[symbol], "beta": beta.get(symbol), "shares": row["shares"],
                      "option_delta": row["option_delta"], "net_delta": net, "dollar_delta": dollars,
                      "benchmark_dollar_delta": None if dollars is None or symbol not in beta else dollars * beta[symbol],
                      "gamma_per_1pct": row["gamma_per_1pct"], "vega": row["vega"], "theta": row["theta"]})
    keys = ("dollar_delta", "benchmark_dollar_delta", "gamma_per_1pct", "vega", "theta")
    return {"rows": table, "total": {k: None if any(r[k] is None for r in table) else sum(r[k] for r in table) for k in keys}}


def settle(legs: list[Leg], spots: dict[str, float], expiry: date, borderline: float
           ) -> tuple[list[Leg], list[dict[str, Any]], float]:
    """Settle `expiry` at current prices: in-the-money options become shares, the rest expire.

    Returns the book after settlement, every short leg in or near the money (`certain` when in
    the money now, else `possible` within `borderline`), and the cash the certain ones move.
    """
    after: list[Leg] = []
    flagged, cash = [], 0.0
    shares: defaultdict[str, float] = defaultdict(float)
    for leg in legs:
        if not leg.option or leg.expiry != expiry:
            after.append(leg)
            continue
        spot = spots[leg.symbol]
        sign = 1 if leg.right == "CALL" else -1
        moneyness = sign * (spot - leg.strike) / leg.strike
        delivered = leg.quantity * 100 * sign  # long call / short put take shares
        if moneyness > 0:
            shares[leg.symbol] += delivered
            cash -= delivered * leg.strike
        if leg.quantity < 0 and moneyness > -borderline:
            flagged.append({"symbol": leg.symbol, "right": leg.right, "strike": leg.strike, "quantity": leg.quantity,
                            "spot": spot, "moneyness": moneyness, "state": "certain" if moneyness > 0 else "possible",
                            "shares": delivered, "cash": -delivered * leg.strike})
    held: defaultdict[str, float] = defaultdict(float)
    for leg in after:
        if not leg.option:
            held[leg.symbol] += leg.quantity
    after = [x for x in after if x.option] + [Leg(s, held[s] + shares[s]) for s in sorted(set(held) | set(shares))
                                              if held[s] + shares[s]]
    return after, flagged, cash


def coverage(book: list[Leg], flagged: list[dict[str, Any]], expiry: date) -> list[dict[str, Any]]:
    """Shares held after settlement against short calls at `expiry`, with and without the possible assignments."""
    shares: defaultdict[str, float] = defaultdict(float)
    for x in book:
        if not x.option:
            shares[x.symbol] += x.quantity
    possible: defaultdict[str, float] = defaultdict(float)
    for f in flagged:
        if f["state"] == "possible":
            possible[f["symbol"]] += f["shares"]
    calls: defaultdict[str, float] = defaultdict(float)
    for leg in book:
        if leg.option and leg.expiry == expiry and leg.right == "CALL" and leg.quantity < 0:
            calls[leg.symbol] += -leg.quantity * 100
    rows = []
    for symbol in sorted(set(shares) | set(possible) | set(calls)):
        held, maybe = shares[symbol], shares[symbol] + possible[symbol]
        rows.append({"symbol": symbol, "shares": held, "shares_if_possible": maybe, "short_calls": calls[symbol],
                     "uncovered": max(held - calls[symbol], 0), "uncovered_if_possible": max(maybe - calls[symbol], 0)})
    return rows


def _contracts(book: list[Leg], marks: dict[str, float], now: datetime, rate: float, iv_multiplier: float
               ) -> tuple[dict[str, float], dict[str, OptionContract], dict[str, float]]:
    positions, contracts, prices = {}, {}, dict(marks)
    for i, leg in enumerate(book):
        if not leg.option:
            positions[leg.symbol] = positions.get(leg.symbol, 0) + leg.quantity
            continue
        assert leg.expiry is not None and leg.right is not None
        key = f"{leg.symbol}:{leg.expiry}:{leg.right}:{leg.strike}:{i}"
        contracts[key] = OptionContract(key, leg.symbol, OptionRight(leg.right), leg.strike, close_of(leg.expiry),
                                        100, "american", "physical")
        positions[key] = leg.quantity
        prices[key] = _price(leg, marks[leg.symbol], now, rate, iv_multiplier)
    return positions, contracts, prices


def fit_margin(book: list[Leg], spots: dict[str, float], now: datetime, rate: float, reported: float,
               leveraged: dict[str, float]) -> tuple[RegTOptionMargin, dict[str, Any]]:
    """The leveraged-ETF factor (1 to 3x the fund's own leverage) that reproduces `reported`."""
    positions, contracts, prices = _contracts(book, spots, now, rate, 1.0)

    def model(scale: float) -> RegTOptionMargin:
        return RegTOptionMargin(stock_fraction=0.25,
                                leverage={s: max(1.0, f * scale) for s, f in leveraged.items()})

    def estimate_at(scale: float) -> float:
        return model(scale).requirement(positions, contracts, prices)

    low, high = 1 / 3, 1.0
    if reported >= estimate_at(high):
        low = high
    elif reported <= estimate_at(low):
        high = low
    for _ in range(40):
        if high - low < 1e-6:
            break
        mid = (low + high) / 2
        low, high = (mid, high) if estimate_at(mid) < reported else (low, mid)
    scale = (low + high) / 2
    fitted = model(scale)
    estimate = fitted.requirement(positions, contracts, prices)
    return fitted, {"scale": scale, "estimate": estimate, "reported": reported, "residual": estimate / reported - 1,
                    "factors": dict(fitted.leverage)}


def scenarios(book: list[Leg], spots: dict[str, float], beta: dict[str, float], shocks: list[float], now: datetime,
              horizon: date, rate: float, iv_up: float, margin: RegTOptionMargin, net_liquidation: float
              ) -> list[dict[str, Any]]:
    def shocked(shock: float) -> dict[str, float]:
        return {s: spots[s] * (1 + max(-0.95, beta[s] * shock)) for s in {leg.symbol for leg in book}}

    def value(prices: dict[str, float], at: datetime, iv_multiplier: float) -> float:
        total = 0.0
        for leg in book:
            total += leg.quantity * (prices[leg.symbol] if not leg.option
                                     else 100 * _price(leg, prices[leg.symbol], at, rate, iv_multiplier))
        return total

    base = value(spots, now, 1.0)
    mid = now + (close_of(horizon) - now) / 2
    rows = []
    for shock in shocks:
        prices, ivm = shocked(shock), (iv_up if shock < 0 else 1.0)
        instant = value(prices, now, ivm) - base
        positions, contracts, marks = _contracts(book, prices, now, rate, ivm)
        requirement = margin.requirement(positions, contracts, marks)
        rows.append({"shock": shock, "pnl_instant": instant, "pnl_mid": value(prices, mid, ivm) - base,
                     "pnl_expiry": value(prices, close_of(horizon) - timedelta(seconds=1), 1.0) - base,
                     "maintenance": requirement, "excess_liquidity": net_liquidation + instant - requirement})
    return rows


def liquidation_shock(book: list[Leg], spots: dict[str, float], beta: dict[str, float], now: datetime, horizon: date,
                      rate: float, iv_up: float, margin: RegTOptionMargin, net_liquidation: float) -> float | None:
    """The benchmark drop at which excess liquidity after an instant shock reaches zero, if within 50%."""
    def excess(shock: float) -> float:
        return scenarios(book, spots, beta, [shock], now, horizon, rate, iv_up, margin, net_liquidation)[0]["excess_liquidity"]
    if excess(0.0) <= 0:
        return 0.0
    if excess(-0.5) > 0:
        return None
    low, high = -0.5, 0.0
    for _ in range(30):
        mid = (low + high) / 2
        low, high = (mid, high) if excess(mid) <= 0 else (low, mid)
    return high


Listed = dict[float, dict[str, Any]]  # strike -> a chain row (bid, ask, implied_volatility, delta)


def listed_quotes(chains: dict[str, Any]) -> dict[tuple[str, str, str], Listed]:
    """Collected chain rows by (underlying, expiry, right), for quoting reference tables."""
    out: dict[tuple[str, str, str], Listed] = {}
    for symbol, rows in chains.items():
        for row in rows if isinstance(rows, list) else []:
            out.setdefault((symbol, row["expiry"], row["right"]), {})[float(row["strike"])] = row
    return out


def _quote(right: OptionRight, spot: float, strike: float, t: float, rate: float, iv: float,
           row: dict[str, Any] | None) -> tuple[float, float, float, str]:
    """Premium per contract, delta, IV and source: the market mid when quoted, else Black-Scholes."""
    if row and row.get("bid") and row.get("ask") and row.get("implied_volatility") and row.get("delta") is not None:
        return 50 * (float(row["bid"]) + float(row["ask"])), float(row["delta"]), float(row["implied_volatility"]), "market"
    return (100 * black_scholes(right, spot, strike, t, rate, iv), bsm_greeks(right, spot, strike, t, rate, iv).delta,
            iv, "model")


def put_hedges(spot: float, iv: float, beta: float, expiry: date, now: datetime, rate: float,
               offsets: list[float], listed: Listed | None = None) -> list[dict[str, Any]]:
    rows, t = [], years_to(expiry, now)
    for offset in offsets:
        target = spot * (1 - offset)
        nearest = min(listed, key=lambda k: abs(k - target)) if listed else None
        # A listed strike far from the target answers a different question: price the target instead.
        strike = nearest if nearest is not None and abs(nearest - target) <= 0.005 * spot else round(target)
        premium, delta, _, source = _quote(OptionRight.PUT, spot, strike, t, rate, iv, (listed or {}).get(strike))
        rows.append({"strike": strike, "offset": 1 - strike / spot, "premium": premium, "delta": delta,
                     "benchmark_dollar_delta_offset": -delta * 100 * spot * beta, "source": source})
    return rows


def hedge_plan(book: list[Leg], spots: dict[str, float], beta: dict[str, float], now: datetime, horizon: date,
               rate: float, iv_up: float, margin: RegTOptionMargin, net_liquidation: float,
               benchmark_dollar_delta: float, hedges: dict[str, list[dict[str, Any]]], cuts: list[float],
               ivs: dict[str, float]) -> list[dict[str, Any]]:
    """Index puts that cut the book's benchmark dollar delta by each fraction in `cuts`.

    Under Reg-T a long option has no loan value (the margin model charges its full premium), so a
    put on another underlying protects profit and loss without moving the liquidation line out;
    one that pairs with the book's own short puts on the same underlying does, through spread netting.

    For every hedge instrument and every reference put: the contracts that remove that share of
    the benchmark exposure, their cost, and the forced-liquidation move once they are held with
    the book (bought now, paid from cash, valued like every other leg).
    """
    rows = []
    for symbol, candidates in hedges.items():
        for put in candidates:
            per_contract = put["benchmark_dollar_delta_offset"]
            if per_contract <= 0 or abs(put["delta"]) < 0.05:  # sizing a lottery ticket by delta means nothing
                continue
            for cut in cuts:
                contracts = ceil(cut * benchmark_dollar_delta / per_contract)
                if contracts <= 0:
                    continue
                leg = Leg(symbol, contracts, horizon, float(put["strike"]), "PUT", ivs[symbol],
                          put["delta"])
                hedged = [*book, leg]
                shock = liquidation_shock(hedged, spots, beta, now, horizon, rate, iv_up, margin, net_liquidation)
                stress = scenarios(hedged, spots, beta, [-0.05, -0.10], now, horizon, rate, iv_up, margin, net_liquidation)
                rows.append({"symbol": symbol, "strike": put["strike"], "offset": put["offset"], "cut": cut,
                             "pnl_down_5": stress[0]["pnl_instant"], "pnl_down_10": stress[1]["pnl_instant"],
                             "contracts": contracts, "cost": contracts * put["premium"],
                             "remaining_benchmark_dollar_delta": benchmark_dollar_delta - contracts * per_contract,
                             "liquidation_shock": shock, "source": put.get("source", "model")})
    return rows


def covered_calls(spot: float, iv: float, cost: float | None, expiry: date, now: datetime, rate: float,
                  strikes: list[float], listed: Listed | None = None) -> list[dict[str, Any]]:
    t, rows = years_to(expiry, now), []
    if listed:
        strikes = sorted(k for k in listed if k >= spot * 0.98)[:8] or strikes
    for strike in strikes:
        premium, delta, vol, source = _quote(OptionRight.CALL, spot, strike, t, rate, iv, (listed or {}).get(strike))
        d2 = (log(spot / strike) + (rate - vol * vol / 2) * t) / (vol * sqrt(t))
        rows.append({"strike": strike, "premium": premium, "delta": delta, "probability_called": 0.5 * (1 + erf(d2 / sqrt(2))),
                     "if_called_vs_cost": None if cost is None else (strike - cost) * 100, "source": source})
    return rows


def _strike_grid(spot: float) -> list[float]:
    """Eight strikes from the money upward, on the step listed options usually use."""
    step = 0.5 if spot < 25 else 1.0 if spot < 100 else 5.0
    first = round(spot / step) * step
    return [first + i * step for i in range(8)]


def analyse(collected: dict[str, Any], events: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    now = datetime.fromisoformat(collected["as_of"])
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("portfolio_aware_as_of_required")
    rate, benchmark = config.get("rate", 0.04), config.get("benchmark", "QQQ")
    legs = [Leg(**{**p, "expiry": date.fromisoformat(p["expiry"]) if p.get("expiry") else None})
            for p in collected["positions"] if p["quantity"]]
    spots = collected["spots"]
    beta = {s: v["beta"] for s, v in betas(collected["history"], benchmark).items()}
    account = collected["account"]
    today = now.astimezone(NEW_YORK).date()
    expiries = sorted({x.expiry for x in legs if x.expiry and x.expiry >= today})
    front = expiries[0] if expiries else None
    settles_today = front == today and now < close_of(today)
    book, flagged, cash = (settle(legs, spots, front, config.get("borderline", 0.015))
                           if settles_today and front else (legs, [], 0.0))
    horizon = next((e for e in expiries if e > today), front) if settles_today else front
    margin, fit = fit_margin(legs, spots, now, rate, account["MaintMarginReq"], config.get("leveraged", {}))
    net = account["NetLiquidation"]
    shocks = config.get("shocks", [-0.10, -0.07, -0.05, -0.03, -0.02, 0.0, 0.02, 0.03, 0.05])
    iv_up = config.get("iv_up", 1.3)
    report: dict[str, Any] = {
        "as_of": collected["as_of"], "benchmark": benchmark, "account": account,
        "exposures": exposures(legs, spots, beta), "betas": beta,
        "settlement": {"expiry": str(front) if settles_today else None, "flagged": flagged, "certain_cash": cash,
                       "possible_cash": sum(f["cash"] for f in flagged if f["state"] == "possible")},
        "horizon": str(horizon) if horizon else None, "margin_fit": fit, "events": events,
    }
    if horizon:
        hedges = [h for h in config.get("hedges", ["QQQ"]) if h in spots]
        if not ({leg.symbol for leg in legs} | set(hedges)) <= beta.keys():
            raise ValueError("portfolio_historical_beta_unavailable")
        report["coverage"] = coverage(book, flagged, horizon)
        report["scenarios"] = scenarios(book, spots, beta, shocks, now, horizon, rate, iv_up, margin, net)
        report["liquidation_shock"] = liquidation_shock(book, spots, beta, now, horizon, rate, iv_up, margin, net)
        if settles_today:  # until the close the book still holds today's short options
            report["liquidation_shock_before_settlement"] = liquidation_shock(
                legs, spots, beta, now, horizon, rate, iv_up, margin, net)
        ivs = {x.symbol: x.iv for x in legs if x.option and x.expiry == horizon and x.iv}
        hedge_iv = config.get("hedge_iv", {})
        listed = listed_quotes(collected.get("chains") or {})
        def market_iv(symbol: str) -> float | None:
            quoted = [float(r["implied_volatility"]) for r in listed.get((symbol, str(horizon), "PUT"), {}).values()
                      if r.get("implied_volatility")]
            return statistics.median(quoted) if quoted else None

        hedge_ivs = {h: hedge_iv.get(h) or market_iv(h) or ivs.get(h) or 0.25 for h in hedges}
        report["put_hedges"] = {h: put_hedges(spots[h], hedge_ivs[h], beta[h],
                                              horizon, now, rate, config.get("hedge_offsets", [0.0, 0.03, 0.05]),
                                              listed.get((h, str(horizon), "PUT")))
                                for h in hedges}
        benchmark_delta = report["exposures"]["total"]["benchmark_dollar_delta"]
        if benchmark_delta is None:
            report["hedge_plan_unavailable"] = "option_delta_unavailable"
        else:
            report["hedge_plan"] = hedge_plan(book, spots, beta, now, horizon, rate, iv_up, margin, net,
                                              benchmark_delta, report["put_hedges"],
                                              config.get("hedge_cuts", [0.25, 0.5]), hedge_ivs)
        costs = {x.symbol: x.average_cost for x in legs if not x.option and x.average_cost}
        report["covered_calls"] = {r["symbol"]: covered_calls(spots[r["symbol"]], ivs.get(r["symbol"]) or 0.5,
                                                              costs.get(r["symbol"]), horizon, now, rate,
                                                              _strike_grid(spots[r["symbol"]]),
                                                              listed.get((r["symbol"], str(horizon), "CALL")))
                                   for r in report["coverage"] if r["uncovered_if_possible"] and r["symbol"] in spots}
    return report


def _money(value: float | None, spec: str = ",.0f") -> str:
    return "—" if value is None else format(value, spec)


def _fmt_shock(shock: float | None) -> str:
    return "beyond -50%" if shock is None else f"{shock:+.1%}"


def render(report: dict[str, Any]) -> str:
    """The report as Markdown, for reading."""
    a, e = report["account"], report["exposures"]
    lines = [f"# Portfolio risk {report['as_of']}", "",
             f"Net liquidation {_money(a.get('NetLiquidation'))} · cash {_money(a.get('TotalCashValue'))} · "
             f"maintenance {_money(a.get('MaintMarginReq'))} · excess liquidity {_money(a.get('ExcessLiquidity'))}", "",
             f"Dollar delta {_money(e['total']['dollar_delta'])} · in {report['benchmark']} terms "
             f"{_money(e['total']['benchmark_dollar_delta'])} · gamma per 1% {_money(e['total']['gamma_per_1pct'])} · "
             f"theta per day {_money(e['total']['theta'])}", "",
             "| symbol | spot | beta | shares | option Δ | $Δ | benchmark $Δ |", "|---|---|---|---|---|---|---|"]
    for r in e["rows"]:
        lines.append(f"| {r['symbol']} | {r['spot']:.2f} | {_money(r['beta'], '.2f')} | {r['shares']:.0f} | "
                     f"{_money(r['option_delta'], '.0f')} | {_money(r['dollar_delta'])} | {_money(r['benchmark_dollar_delta'])} |")
    s = report["settlement"]
    if s["expiry"]:
        lines += ["", f"## Settling today ({s['expiry']})", "",
                  f"Certain cash {_money(s['certain_cash'])} · possible {_money(s['possible_cash'])}", "",
                  "| symbol | right | strike | qty | spot | state | shares | cash |", "|---|---|---|---|---|---|---|---|"]
        lines += [f"| {f['symbol']} | {f['right']} | {f['strike']} | {f['quantity']:.0f} | {f['spot']:.2f} | {f['state']} | "
                  f"{f['shares']:+.0f} | {_money(f['cash'])} |" for f in s["flagged"]]
    if report.get("coverage"):
        lines += ["", f"## Covered-call cover at {report['horizon']}", "",
                  "| symbol | shares | if possible | short calls | uncovered | if possible |", "|---|---|---|---|---|---|"]
        lines += [f"| {c['symbol']} | {c['shares']:.0f} | {c['shares_if_possible']:.0f} | {c['short_calls']:.0f} | "
                  f"{c['uncovered']:.0f} | {c['uncovered_if_possible']:.0f} |" for c in report["coverage"]]
    if report.get("scenarios"):
        fit = report["margin_fit"]
        shock = report.get("liquidation_shock")
        lines += ["", f"## Scenarios ({report['benchmark']} move, names by beta)", "",
                  f"Margin model fitted to the broker's maintenance: residual {fit['residual']:+.1%}. "
                  f"Excess liquidity reaches zero at a {report['benchmark']} move of "
                  f"{_fmt_shock(shock)}"
                  + (f" (before today's settlement: {report['liquidation_shock_before_settlement']:+.1%})"
                     if report.get("liquidation_shock_before_settlement") is not None else "") + ".", "",
                  "| move | P&L now | P&L mid-way | P&L at expiry | maintenance | excess liquidity |", "|---|---|---|---|---|---|"]
        lines += [f"| {r['shock']:+.0%} | {_money(r['pnl_instant'])} | {_money(r['pnl_mid'])} | {_money(r['pnl_expiry'])} | "
                  f"{_money(r['maintenance'])} | {_money(r['excess_liquidity'])} |" for r in report["scenarios"]]
    for symbol, rows in (report.get("put_hedges") or {}).items():
        lines += ["", f"## {symbol} puts to {report['horizon']} (reference)", "",
                  "| strike | below spot | premium | delta | offsets benchmark $Δ | price |", "|---|---|---|---|---|---|"]
        lines += [f"| {r['strike']} | {r['offset']:.1%} | {_money(r['premium'])} | {r['delta']:+.2f} | "
                  f"{_money(r['benchmark_dollar_delta_offset'])} | {r.get('source', 'model')} |" for r in rows]
    if report.get("hedge_plan_unavailable"):
        lines += ["", f"## Hedge plan: unavailable ({report['hedge_plan_unavailable']})"]
    if report.get("hedge_plan"):
        lines += ["", f"## Hedge plan: puts to {report['horizon']} that cut the {report['benchmark']}-terms delta "
                  "(reference, not advice)", "",
                  "Under Reg-T a long option has no loan value: these puts protect profit and loss, but their gains "
                  "do not count toward margin and their premium leaves cash, so the forced-liquidation line moves out "
                  "only where they pair with short options on the same underlying (or once they are sold).", "",
                  "| put | cut | contracts | cost | remaining benchmark $Δ | P&L at -5% | P&L at -10% | "
                  "liquidation move after | price |", "|---|---|---|---|---|---|---|---|---|"]
        unhedged = {round(r["shock"], 2): r["pnl_instant"] for r in report.get("scenarios", [])}
        lines += [f"| unhedged | | | | {_money(report['exposures']['total']['benchmark_dollar_delta'])} | "
                  f"{_money(unhedged.get(-0.05))} | {_money(unhedged.get(-0.1))} | "
                  f"{_fmt_shock(report.get('liquidation_shock'))} | |"]
        lines += [f"| {r['symbol']} {r['strike']} ({r['offset']:.1%} below) | {r['cut']:.0%} | {r['contracts']} | "
                  f"{_money(r['cost'])} | {_money(r['remaining_benchmark_dollar_delta'])} | "
                  f"{_money(r['pnl_down_5'])} | {_money(r['pnl_down_10'])} | "
                  f"{_fmt_shock(r['liquidation_shock'])} | {r['source']} |" for r in report["hedge_plan"]]
    for symbol, rows in (report.get("covered_calls") or {}).items():
        lines += ["", f"## {symbol} calls to {report['horizon']} for uncovered shares (reference)", "",
                  "| strike | premium | delta | P(called) | if called vs cost /100sh | price |", "|---|---|---|---|---|---|"]
        lines += [f"| {r['strike']} | {_money(r['premium'])} | {r['delta']:.2f} | {r['probability_called']:.0%} | "
                  f"{_money(r['if_called_vs_cost'])} | {r.get('source', 'model')} |" for r in rows]
    if report.get("whatif"):
        lines += ["", "## Broker what-if (margin preview)", ""]
        lines += [f"- {o['action']} {o['quantity']} {o['symbol']} {o.get('expiry', '')} {o.get('strike', '')} "
                  f"{o.get('right', '')}: {o['status']}"
                  + (f", maintenance change {_money(o.get('maintMarginChange'))}" if o["status"] == "computed" else "")
                  for o in report["whatif"]]
    events = report.get("events") or {}
    if events.get("unavailable"):
        lines += ["", f"## Events before the horizon: unavailable ({events['unavailable']})"]
    if events.get("earnings") or events.get("macro"):
        lines += ["", "## Events before the horizon", ""]
        lines += [f"- {x['date']} earnings: {x['symbol']}{' (held)' if x.get('held') else ''}"
                  for x in events.get("earnings", [])]
        lines += [f"- {x['date']} {x['title']}" for x in events.get("macro", [])]
    lines += ["", "Research estimate (Black-Scholes at each contract's IV, Reg-T model fitted to the broker); "
              "not the broker's risk engine and not advice. Reference puts may assume 25% IV without configured, "
              "quoted or held-option IV; reference calls may assume 50% without a usable quote or held-option IV. "
              "These defaults are model inputs, not observed volatility."]
    return "\n".join(lines) + "\n"


__all__ = ["Leg", "analyse", "betas", "coverage", "covered_calls", "exposures", "fit_margin", "hedge_plan",
           "liquidation_shock",
           "put_hedges", "render", "scenarios", "settle"]
