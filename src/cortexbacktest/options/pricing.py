"""Vanilla pricing benchmarks; prices and Greeks are per underlying unit.

Continuous rates/yields, ACT/365 time supplied in years. No discrete dividends.
Greeks: vega/rho per 1.0 change, theta per year (calendar-time decay).
"""
from dataclasses import dataclass
from math import erf, exp, isfinite, log, pi, sqrt
from enum import StrEnum

import numpy as np

class OptionRight(StrEnum):
    """Call or put. The value is the stored word, so it compares equal to the
    strings in chain files and frames."""

    CALL = "CALL"
    PUT = "PUT"

    @property
    def sign(self) -> int:
        """+1 for a call, -1 for a put: the payoff is max(sign * (spot - strike), 0)."""
        return 1 if self is OptionRight.CALL else -1


def option_right(value: str) -> OptionRight:
    """The right `value` names; anything but CALL or PUT is refused."""
    try:
        return OptionRight(value)
    except ValueError:
        raise ValueError("right must be CALL or PUT") from None


def _validate(right: OptionRight, spot: float, strike: float, time: float,
              rate: float, volatility: float, dividend: float) -> int:
    sign = option_right(right).sign
    if not all(isfinite(x) for x in (spot, strike, time, rate, volatility, dividend)):
        raise ValueError("pricing inputs must be finite")
    if spot <= 0 or strike <= 0 or time < 0 or volatility < 0:
        raise ValueError("spot/strike must be positive; time/volatility nonnegative")
    return sign


def _cdf(x: float) -> float:
    return 0.5 * (1 + erf(x / sqrt(2)))


def black_scholes(right: OptionRight, spot: float, strike: float, time: float,
                  rate: float, volatility: float, dividend: float = 0) -> float:
    """European BSM with continuous dividend yield; handles zero time/volatility."""
    sign = _validate(right, spot, strike, time, rate, volatility, dividend)
    a, b = spot * exp(-dividend * time), strike * exp(-rate * time)
    if time == 0 or volatility == 0:
        return max(sign * (a - b), 0.0)
    v = volatility * sqrt(time)
    d1 = (log(spot / strike) + (rate - dividend) * time) / v + v / 2
    return max(sign * (a * _cdf(sign * d1) - b * _cdf(sign * (d1 - v))), 0.0)


def black76(right: OptionRight, forward: float, strike: float, time: float,
            rate: float, volatility: float) -> float:
    """European premium-paid forward option; not futures-style daily settlement."""
    return black_scholes(right, forward, strike, time, rate, volatility, rate)


@dataclass(frozen=True, slots=True)
class OptionGreeks:
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float


def bsm_greeks(right: OptionRight, spot: float, strike: float, time: float,
               rate: float, volatility: float, dividend: float = 0) -> OptionGreeks:
    sign = _validate(right, spot, strike, time, rate, volatility, dividend)
    if time == 0 or volatility == 0:
        raise ValueError("Greeks require positive time and volatility (singular boundary)")
    v = volatility * sqrt(time)
    d1 = (log(spot / strike) + (rate - dividend) * time) / v + v / 2
    d2 = d1 - v
    density = exp(-d1 * d1 / 2) / sqrt(2 * pi)
    dq, dr = exp(-dividend * time), exp(-rate * time)
    return OptionGreeks(
        sign * dq * _cdf(sign * d1), dq * density / (spot * v),
        spot * dq * density * sqrt(time),
        -spot * dq * density * volatility / (2 * sqrt(time))
        - sign * rate * strike * dr * _cdf(sign * d2)
        + sign * dividend * spot * dq * _cdf(sign * d1),
        sign * strike * time * dr * _cdf(sign * d2),
    )


def implied_volatility(price: float, right: OptionRight, spot: float, strike: float,
                       time: float, rate: float, dividend: float = 0, *,
                       max_volatility: float = 8, tolerance: float = 1e-10) -> float:
    """Bracketed European IV. No HV fallback; impossible/unidentified prices fail."""
    _validate(right, spot, strike, time, rate, 0, dividend)
    if (not all(isfinite(x) for x in (price, max_volatility, tolerance))
            or price < 0 or max_volatility <= 0 or tolerance <= 0 or time <= 0):
        raise ValueError("IV needs finite price, positive time/bracket/tolerance")
    lower = black_scholes(right, spot, strike, time, rate, 0, dividend)
    upper = (spot * exp(-dividend * time) if right == "CALL"
             else strike * exp(-rate * time))
    if price < lower or price >= upper:
        raise ValueError("price outside finite-volatility European arbitrage bounds")
    if price == lower:
        return 0.0
    if black_scholes(right, spot, strike, time, rate, max_volatility, dividend) < price:
        raise ValueError("IV exceeds configured bracket")
    lo, hi = 0.0, max_volatility
    for _ in range(200):
        mid = (lo + hi) / 2
        value = black_scholes(right, spot, strike, time, rate, mid, dividend)
        if value < price:
            lo = mid
        else:
            hi = mid
        if hi - lo <= tolerance:
            return (lo + hi) / 2
    raise ArithmeticError("IV did not converge")


def crr_price(right: OptionRight, spot: float, strike: float, time: float,
              rate: float, volatility: float, dividend: float = 0, *,
              steps: int = 500, american: bool = True) -> float:
    """CRR backward induction, O(steps) memory. Early exercise on tree dates."""
    sign = _validate(right, spot, strike, time, rate, volatility, dividend)
    if isinstance(steps, bool) or not isinstance(steps, int) or not 1 <= steps <= 20000:
        raise ValueError("steps must be an integer in [1, 20000]")
    if time == 0:
        return max(sign * (spot - strike), 0.0)
    if volatility == 0:
        dates = np.linspace(0, time, steps + 1) if american else np.array([time])
        payoffs = np.exp(-rate * dates) * np.maximum(
            sign * (spot * np.exp((rate - dividend) * dates) - strike), 0)
        return float(payoffs.max())
    dt = time / steps
    up = exp(volatility * sqrt(dt))
    down = 1 / up
    probability = (exp((rate - dividend) * dt) - down) / (up - down)
    return _binomial(sign, spot, strike, up, down, probability, exp(-rate * dt), steps, american)


def _binomial(sign: int, spot: float, strike: float, up: float, down: float, probability: float,
              discount: float, steps: int, american: bool) -> float:
    """Backward induction on a recombining tree; early exercise checked at every node."""
    if not 0 <= probability <= 1:
        raise ValueError("invalid tree probability; increase steps or use another model")
    j = np.arange(steps + 1)
    nodes = spot * np.exp(j * log(up) + (steps - j) * log(down))
    values = np.maximum(sign * (nodes - strike), 0)
    for _ in range(steps):
        values = discount * ((1 - probability) * values[:-1] + probability * values[1:])
        if american:
            nodes = nodes[:-1] / down
            values = np.maximum(values, np.maximum(sign * (nodes - strike), 0))
    result = float(values[0])
    if not isfinite(result):
        raise ArithmeticError("tree overflow; reduce input scale")
    return result
