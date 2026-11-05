"""Numerical option pricing: trees, finite differences and Monte Carlo.

Same conventions as `pricing`: continuous rate and dividend yield, time in years,
prices per underlying unit. `pricing.black_scholes` is the European benchmark and
`pricing.crr_price` the plain binomial tree; these methods add faster convergence,
a PDE solver and path dependence. Trees and the PDE solver delegate zero time or
volatility to `crr_price`; Monte Carlo requires positive time.

- Leisen-Reimer binomial: Peizer-Pratt probabilities centred on the strike, odd steps;
  European error falls like 1/n^2 instead of CRR's oscillating 1/n.
- Trinomial (Boyle): up / flat / down per step; smoother than a binomial tree.
- Crank-Nicolson on log price, with fully implicit Rannacher start-up steps that damp the
  payoff kink; American exercise by projection onto the payoff after each step.
- Monte Carlo on exact GBM paths with antithetic draws: European, arithmetic Asian with
  equally spaced fixings (the last at expiry), and American by Longstaff-Schwartz
  regression of continuation values on a cubic in moneyness over in-the-money paths.
  LSM prices the Bermudan option with `steps` exercise dates and is biased slightly low
  (a quadratic basis was 0.02 low on the standard put; cubic is within noise); every
  estimate carries its standard error.
"""
from dataclasses import dataclass
from math import exp, log, sqrt

import numpy as np

from .pricing import OptionRight, _binomial, _validate, crr_price


@dataclass(frozen=True, slots=True)
class MonteCarloPrice:
    price: float
    std_error: float


def _count(value: int, name: str, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in [{low}, {high}]")
    return value


def leisen_reimer_price(right: OptionRight, spot: float, strike: float, time: float,
                        rate: float, volatility: float, dividend: float = 0, *,
                        steps: int = 201, american: bool = True) -> float:
    sign = _validate(right, spot, strike, time, rate, volatility, dividend)
    steps = _count(steps, "steps", 1, 20000)
    if time == 0 or volatility == 0:
        return crr_price(right, spot, strike, time, rate, volatility, dividend, steps=steps, american=american)
    n = steps if steps % 2 else steps + 1
    v = volatility * sqrt(time)
    d1 = (log(spot / strike) + (rate - dividend) * time) / v + v / 2

    def peizer_pratt(z: float) -> float:
        scale = (z / (n + 1 / 3 + 0.1 / (n + 1))) ** 2 * (n + 1 / 6)
        return 0.5 + np.copysign(0.5, z) * sqrt(1 - exp(-scale))

    probability, probability_bar = peizer_pratt(d1 - v), peizer_pratt(d1)
    dt = time / n
    growth = exp((rate - dividend) * dt)
    up = growth * probability_bar / probability
    down = (growth - probability * up) / (1 - probability)
    return _binomial(sign, spot, strike, up, down, probability, exp(-rate * dt), n, american)


def trinomial_price(right: OptionRight, spot: float, strike: float, time: float,
                    rate: float, volatility: float, dividend: float = 0, *,
                    steps: int = 500, american: bool = True) -> float:
    sign = _validate(right, spot, strike, time, rate, volatility, dividend)
    steps = _count(steps, "steps", 1, 20000)
    if time == 0 or volatility == 0:
        return crr_price(right, spot, strike, time, rate, volatility, dividend, steps=steps, american=american)
    dt = time / steps
    half = volatility * sqrt(dt / 2)
    drift = exp((rate - dividend) * dt / 2)
    span = exp(half) - exp(-half)
    up, down = ((drift - exp(-half)) / span) ** 2, ((exp(half) - drift) / span) ** 2
    middle = 1 - up - down
    if min(up, down, middle) < 0:
        raise ValueError("invalid trinomial probabilities; increase steps")
    nodes = spot * np.exp(2 * half * np.arange(-steps, steps + 1))
    values = np.maximum(sign * (nodes - strike), 0)
    discount = exp(-rate * dt)
    for _ in range(steps):
        values = discount * (down * values[:-2] + middle * values[1:-1] + up * values[2:])
        if american:
            nodes = nodes[1:-1]
            values = np.maximum(values, np.maximum(sign * (nodes - strike), 0))
    return float(values[0])


def _solve_tridiagonal(bands: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    "NumPy-only tridiagonal elimination with adjacent-row partial pivoting."
    n = len(rhs)
    diagonal = bands[1].tolist()
    upper, lower = bands[0, 1:].tolist(), bands[2, :-1].tolist()
    second = [0.] * max(0, n - 2)
    result = rhs.tolist()
    for i in range(n - 1):
        if abs(diagonal[i]) >= abs(lower[i]):
            if diagonal[i] == 0:
                raise np.linalg.LinAlgError("singular tridiagonal system")
            factor = lower[i] / diagonal[i]
            diagonal[i + 1] -= factor * upper[i]
            result[i + 1] -= factor * result[i]
        else:
            factor = diagonal[i] / lower[i]
            diagonal[i], diagonal[i + 1], upper[i] = (
                lower[i], upper[i] - factor * diagonal[i + 1], diagonal[i + 1])
            if i < n - 2:
                second[i] = upper[i + 1]
                upper[i + 1] *= -factor
            result[i], result[i + 1] = result[i + 1], result[i] - factor * result[i + 1]
    if diagonal[-1] == 0:
        raise np.linalg.LinAlgError("singular tridiagonal system")
    result[-1] /= diagonal[-1]
    for i in range(n - 2, -1, -1):
        future = upper[i] * result[i + 1]
        if i < n - 2:
            future += second[i] * result[i + 2]
        result[i] = (result[i] - future) / diagonal[i]
    values = np.asarray(result, dtype=float)
    if not np.isfinite(values).all():
        raise np.linalg.LinAlgError("nonfinite tridiagonal solution")
    return values


def crank_nicolson_price(right: OptionRight, spot: float, strike: float, time: float,
                         rate: float, volatility: float, dividend: float = 0, *,
                         space_steps: int = 400, time_steps: int = 400, american: bool = True,
                         width: float = 6.0) -> float:
    """Solve the Black-Scholes PDE in log price, centred on the spot (an even grid puts it on a node)."""
    sign = _validate(right, spot, strike, time, rate, volatility, dividend)
    space_steps = _count(space_steps, "space_steps", 20, 20000)
    time_steps = _count(time_steps, "time_steps", 4, 20000)
    if time == 0 or volatility == 0:
        return crr_price(right, spot, strike, time, rate, volatility, dividend, american=american)
    m = space_steps + space_steps % 2
    v = volatility * sqrt(time)
    half_width = max(width * v, abs(log(strike / spot)) + 2 * v)
    prices = spot * np.exp(np.linspace(-half_width, half_width, m + 1))
    dx = 2 * half_width / m
    payoff = np.maximum(sign * (prices - strike), 0)
    diffusion, drift = volatility ** 2 / 2, rate - dividend - volatility ** 2 / 2
    lower = diffusion / dx ** 2 - drift / (2 * dx)
    centre = -2 * diffusion / dx ** 2 - rate
    upper = diffusion / dx ** 2 + drift / (2 * dx)
    dt = time / time_steps
    values = payoff
    for k in range(time_steps):
        theta = 1.0 if k < 2 else 0.5  # Rannacher: implicit steps first, then Crank-Nicolson
        tau = (k + 1) * dt
        # Far boundaries: deep in the money is the discounted forward intrinsic value
        # (immediate intrinsic if early exercise is worth more); far out of the money is zero.
        edge = sign * (prices[[0, -1]] * exp(-dividend * tau) - strike * exp(-rate * tau))
        if american:
            edge = np.maximum(edge, sign * (prices[[0, -1]] - strike))
        new_low, new_high = np.maximum(edge, 0)
        inner = values[1:-1]
        rhs = inner + (1 - theta) * dt * (lower * values[:-2] + centre * inner + upper * values[2:])
        rhs[0] += theta * dt * lower * new_low
        rhs[-1] += theta * dt * upper * new_high
        bands = np.zeros((3, m - 1))
        bands[0, 1:] = -theta * dt * upper
        bands[1, :] = 1 - theta * dt * centre
        bands[2, :-1] = -theta * dt * lower
        values = np.concatenate([[new_low], _solve_tridiagonal(bands, rhs), [new_high]])
        if american:
            values = np.maximum(values, payoff)
    return float(values[m // 2])


def _paths(spot: float, time: float, rate: float, volatility: float, dividend: float,
           paths: int, fixings: int, seed: int | None) -> np.ndarray:
    """Prices at `fixings` equally spaced dates; the second half of the rows mirror the first."""
    draws = np.random.default_rng(seed).standard_normal((paths // 2, fixings))
    dt = time / fixings
    steps = (rate - dividend - volatility ** 2 / 2) * dt + volatility * sqrt(dt) * np.vstack([draws, -draws])
    return spot * np.exp(np.cumsum(steps, axis=1))


def _estimate(discounted: np.ndarray) -> MonteCarloPrice:
    half = discounted.size // 2
    pairs = (discounted[:half] + discounted[half:]) / 2  # antithetic pairs are the independent draws
    return MonteCarloPrice(float(pairs.mean()), float(pairs.std(ddof=1) / sqrt(half)))


def _mc_inputs(right: OptionRight, spot: float, strike: float, time: float, rate: float,
               volatility: float, dividend: float, paths: int) -> int:
    sign = _validate(right, spot, strike, time, rate, volatility, dividend)
    if _count(paths, "paths", 4, 10_000_000) % 2:
        raise ValueError("paths must be even (antithetic pairs)")
    if time == 0:
        raise ValueError("Monte Carlo needs positive time")
    return sign


def monte_carlo_price(right: OptionRight, spot: float, strike: float, time: float,
                      rate: float, volatility: float, dividend: float = 0, *,
                      paths: int = 100_000, seed: int | None = None) -> MonteCarloPrice:
    """European price from terminal values."""
    return asian_price(right, spot, strike, time, rate, volatility, dividend, fixings=1, paths=paths, seed=seed)


def asian_price(right: OptionRight, spot: float, strike: float, time: float,
                rate: float, volatility: float, dividend: float = 0, *,
                fixings: int = 12, paths: int = 100_000, seed: int | None = None) -> MonteCarloPrice:
    """Fixed-strike arithmetic-average option over equally spaced fixings, the last at expiry."""
    sign = _mc_inputs(right, spot, strike, time, rate, volatility, dividend, paths)
    fixings = _count(fixings, "fixings", 1, 10000)
    average = _paths(spot, time, rate, volatility, dividend, paths, fixings, seed).mean(axis=1)
    return _estimate(exp(-rate * time) * np.maximum(sign * (average - strike), 0))


def longstaff_schwartz_price(right: OptionRight, spot: float, strike: float, time: float,
                             rate: float, volatility: float, dividend: float = 0, *,
                             steps: int = 50, paths: int = 100_000, basis_degree: int = 3,
                             seed: int | None = None) -> MonteCarloPrice:
    """American price: exercise where intrinsic beats the regressed continuation value."""
    sign = _mc_inputs(right, spot, strike, time, rate, volatility, dividend, paths)
    steps = _count(steps, "steps", 1, 10000)
    basis_degree = _count(basis_degree, "basis_degree", 1, 8)
    prices = _paths(spot, time, rate, volatility, dividend, paths, steps, seed)
    discount = exp(-rate * time / steps)
    cash = np.maximum(sign * (prices[:, -1] - strike), 0)
    for t in range(steps - 2, -1, -1):
        cash *= discount
        intrinsic = np.maximum(sign * (prices[:, t] - strike), 0)
        money = np.flatnonzero(intrinsic > 0)
        if money.size > basis_degree + 1:
            x = prices[money, t] / strike
            continuation = np.polyval(np.polyfit(x, cash[money], basis_degree), x)
            exercise = money[intrinsic[money] > continuation]
            cash[exercise] = intrinsic[exercise]
    estimate = _estimate(cash * discount)
    immediate = max(sign * (spot - strike), 0.0)
    return estimate if estimate.price >= immediate else MonteCarloPrice(immediate, 0.0)
