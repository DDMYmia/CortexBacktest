"""One option priced every way this package can, for a calculator page.

`analyse` takes plain numbers and returns plain JSON: each model's price, the
Black-Scholes Greeks, an optional implied volatility from a market premium, and
two grids for charts -- value and Greeks across spot, and profit across spot and
time left. Values on the grids use the reference model: Leisen-Reimer for an
American option, Black-Scholes for a European one.
"""
from dataclasses import asdict
from typing import Any

import numpy as np

from .numerical import (
    asian_price, crank_nicolson_price, leisen_reimer_price, longstaff_schwartz_price,
    monte_carlo_price, trinomial_price,
)
from .pricing import black_scholes, bsm_greeks, crr_price, implied_volatility, option_right

GRID_SPOTS = 41
TIME_LEFT = (1.0, 0.75, 0.5, 0.25, 0.1, 0.0)  # fractions of the time to expiry


def analyse(request: dict[str, Any]) -> dict[str, Any]:
    right = option_right(str(request["right"]).upper())
    spot, strike, time = float(request["spot"]), float(request["strike"]), float(request["time"])
    rate, volatility = float(request["rate"]), float(request["volatility"])
    dividend = float(request.get("dividend", 0))
    american = bool(request.get("american", True))
    seed = int(request.get("seed", 7))
    args = (right, spot, strike, time, rate, volatility, dividend)

    def reference(at_spot: float, at_time: float) -> float:
        inputs = (right, at_spot, strike, at_time, rate, volatility, dividend)
        return leisen_reimer_price(*inputs, steps=101) if american else black_scholes(*inputs)

    exercise = "American" if american else "European"
    european = black_scholes(*args)
    models: list[dict[str, Any]] = [
        {"model": "Black-Scholes", "exercise": "European", "price": european},
        {"model": "Binomial (CRR)", "exercise": exercise, "price": crr_price(*args, american=american)},
        {"model": "Leisen-Reimer", "exercise": exercise, "price": leisen_reimer_price(*args, american=american)},
        {"model": "Trinomial", "exercise": exercise, "price": trinomial_price(*args, american=american)},
        {"model": "Crank-Nicolson", "exercise": exercise, "price": crank_nicolson_price(*args, american=american)},
    ]
    if time > 0:
        simulated = [("Monte Carlo", "European", monte_carlo_price(*args, paths=100_000, seed=seed)),
                     ("Arithmetic Asian (12 fixings)", "Asian", asian_price(*args, paths=100_000, seed=seed))]
        if american:
            simulated.append(("Longstaff-Schwartz", "Bermudan (50 dates)",
                              longstaff_schwartz_price(*args, paths=50_000, seed=seed)))
        models += [{"model": name, "exercise": kind, "price": result.price, "std_error": result.std_error}
                   for name, kind, result in simulated]

    premium = reference(spot, time) if american else european
    # Three standard deviations of the log price by expiry (5% to 50%), always including the strike.
    spread = min(max(3 * volatility * np.sqrt(time), 0.05), 0.5)
    low, high = min(spot * np.exp(-spread), 0.95 * strike), max(spot * np.exp(spread), 1.05 * strike)
    spots = np.linspace(low, high, GRID_SPOTS).tolist()
    sign = right.sign
    curve = []
    for s in spots:
        point = {"spot": s, "value": reference(s, time), "payoff": max(sign * (s - strike), 0.0)}
        if time > 0 and volatility > 0:
            point.update(asdict(bsm_greeks(right, s, strike, time, rate, volatility, dividend)))
        curve.append(point)
    profit = [{"time_left": fraction * time, "spot": s, "profit": reference(s, fraction * time) - premium}
              for fraction in TIME_LEFT for s in spots]

    implied = None
    if request.get("market_price") is not None:
        try:
            implied = implied_volatility(float(request["market_price"]), right, spot, strike, time, rate, dividend)
        except (ValueError, ArithmeticError) as exc:
            implied = str(exc)
    greeks = (asdict(bsm_greeks(*args)) if time > 0 and volatility > 0 else None)
    return {"reference_model": "Leisen-Reimer" if american else "Black-Scholes", "premium": premium,
            # A tree can sit a hair under the closed form; the premium for exercising early is never negative.
            "early_exercise_premium": max(premium - european, 0.0) if american else 0.0,
            "models": models, "greeks": greeks, "implied_volatility": implied,
            "curve": curve, "profit": profit}
