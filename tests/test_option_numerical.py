"""Numerical pricers agree with the closed form, with each other and with known references."""
import numpy as np
import pytest

from cortexbacktest.options import (
    asian_price, black_scholes, crank_nicolson_price, crr_price, leisen_reimer_price,
    longstaff_schwartz_price, monte_carlo_price, trinomial_price,
)

STANDARD = ("PUT", 100, 100, 1, .05, .2)  # the textbook at-the-money put
AMERICAN_PUT = crr_price(*STANDARD, steps=20000)  # 6.0903, the published value


@pytest.mark.parametrize("args", [
    ("CALL", 100, 100, 1, .05, .2), ("PUT", 100, 100, 1, .05, .2),
    ("CALL", 90, 100, .5, .03, .35, .02), ("PUT", 120, 100, 2, .04, .25, .01),
])
def test_european_methods_match_black_scholes(args):
    exact = black_scholes(*args)
    assert leisen_reimer_price(*args, american=False) == pytest.approx(exact, abs=2e-4)
    assert trinomial_price(*args, american=False) == pytest.approx(exact, abs=3e-3)
    assert crank_nicolson_price(*args, american=False) == pytest.approx(exact, abs=3e-3)
    estimate = monte_carlo_price(*args, paths=200_000, seed=11)
    assert abs(estimate.price - exact) < 4 * estimate.std_error


def test_american_put_methods_agree_with_the_reference():
    assert AMERICAN_PUT == pytest.approx(6.0903, abs=5e-4)
    for price in (leisen_reimer_price(*STANDARD), trinomial_price(*STANDARD), crank_nicolson_price(*STANDARD)):
        assert price == pytest.approx(AMERICAN_PUT, abs=4e-3)
        assert price > black_scholes(*STANDARD)  # early exercise is worth something for a put


def test_longstaff_schwartz_prices_the_bermudan_option_it_can_exercise():
    # Exact value with exercise on the same 50 dates, from a 10000-step tree (see module docstring).
    bermudan = 6.0786
    estimate = longstaff_schwartz_price(*STANDARD, steps=50, paths=200_000, seed=7)
    assert abs(estimate.price - bermudan) < 4 * estimate.std_error + .01
    assert estimate.price < AMERICAN_PUT + 4 * estimate.std_error


def test_without_dividends_an_american_call_is_european():
    call = ("CALL", 100, 95, 1, .05, .3)
    exact = black_scholes(*call)
    assert leisen_reimer_price(*call) == pytest.approx(exact, abs=2e-4)
    assert crank_nicolson_price(*call) == pytest.approx(exact, abs=3e-3)
    # With a dividend yield, early exercise of a deep in-the-money call becomes worth something.
    paying = ("CALL", 150, 100, 1, .01, .2, .08)
    assert leisen_reimer_price(*paying) > leisen_reimer_price(*paying, american=False) + .5


def test_asian_average_is_cheaper_and_one_fixing_is_european():
    call = ("CALL", 100, 100, 1, .05, .2)
    asian = asian_price(*call, fixings=12, paths=200_000, seed=3)
    european = monte_carlo_price(*call, paths=200_000, seed=3)
    assert asian.price < european.price - 3  # averaging damps the terminal spread
    assert asian_price(*call, fixings=1, paths=200_000, seed=3) == european


def test_deterministic_inputs_and_invalid_arguments():
    assert leisen_reimer_price("PUT", 50, 100, 1, .1, 0) == 50  # zero volatility: exercise now
    assert trinomial_price("CALL", 120, 100, 0, .05, .2) == 20  # expiry: intrinsic
    for bad in (dict(steps=0), dict(steps=True)):
        with pytest.raises(ValueError):
            leisen_reimer_price(*STANDARD, **bad)
    with pytest.raises(ValueError, match="even"):
        monte_carlo_price(*STANDARD, paths=1001)
    with pytest.raises(ValueError):
        crank_nicolson_price(*STANDARD, space_steps=10)
    with pytest.raises(ValueError):
        black_scholes("PUT", -1, 100, 1, .05, .2)
    assert np.isfinite(crank_nicolson_price("PUT", 100, 300, 1, .05, .2))  # strike far outside +-6 sigma
