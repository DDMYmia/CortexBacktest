from datetime import datetime, timezone
from math import exp

import pytest

from cortexbacktest.options import (
    black76, black_scholes, bsm_greeks, crr_price, implied_volatility,
)


def test_reference_prices_and_greeks():
    assert black_scholes("CALL", 100, 100, 1, .05, .2) == pytest.approx(10.450583572185565)
    assert black_scholes("PUT", 100, 100, 1, .05, .2) == pytest.approx(5.573526022256971)
    greeks = bsm_greeks("CALL", 100, 100, 1, .05, .2)
    assert greeks.delta == pytest.approx(.6368306511756191)
    assert greeks.gamma == pytest.approx(.018762017345846895)
    assert greeks.vega == pytest.approx(37.52403469169379)


@pytest.mark.parametrize("right", ["CALL", "PUT"])
@pytest.mark.parametrize("rate,dividend", [(.05, .02), (-.01, 0), (0, .10)])
def test_greeks_against_independent_finite_differences(right, rate, dividend):
    s, k, t, v, h = 110, 100, .8, .3, 1e-4
    def price(s=s, t=t, v=v, r=rate):
        return black_scholes(right, s, k, t, r, v, dividend)
    g = bsm_greeks(right, s, k, t, rate, v, dividend)
    assert g.delta == pytest.approx((price(s=s+h)-price(s=s-h))/(2*h), abs=1e-7)
    assert g.gamma == pytest.approx((price(s=s+h)-2*price()+price(s=s-h))/h**2, abs=4e-6)
    assert g.vega == pytest.approx((price(v=v+h)-price(v=v-h))/(2*h), rel=1e-6)
    assert g.theta == pytest.approx(-(price(t=t+h)-price(t=t-h))/(2*h), rel=1e-6)
    assert g.rho == pytest.approx((price(r=rate+h)-price(r=rate-h))/(2*h), rel=1e-6)


@pytest.mark.parametrize("s,k,t,r,v,q", [(80,100,.3,.03,.5,.01),(110,100,2,-.01,.2,.04),
                                       (100,100,.001,0,.9,0)])
def test_parity_and_iv_roundtrip(s,k,t,r,v,q):
    call = black_scholes("CALL",s,k,t,r,v,q)
    put = black_scholes("PUT",s,k,t,r,v,q)
    assert call-put == pytest.approx(s*exp(-q*t)-k*exp(-r*t))
    for right, price in (("CALL",call),("PUT",put)):
        assert implied_volatility(price,right,s,k,t,r,q) == pytest.approx(v,abs=1e-8)
    forward = s*exp((r-q)*t)
    assert black76("CALL",forward,k,t,r,v) == pytest.approx(call)


def test_tree_convergence_and_early_exercise():
    euro = black_scholes("PUT",100,100,1,.05,.2)
    assert crr_price("PUT",100,100,1,.05,.2,steps=1000,american=False) == pytest.approx(euro,abs=.003)
    assert crr_price("PUT",100,100,1,.05,.2,steps=1000) == pytest.approx(6.089595,abs=.003)
    assert crr_price("CALL",100,100,1,.05,.2,steps=1000) == pytest.approx(
        black_scholes("CALL",100,100,1,.05,.2),abs=.003)
    assert crr_price("PUT",50,100,1,.1,0) == 50
    assert crr_price("PUT",50,100,1,.1,0,american=False) < 50


def test_singular_and_invalid_cases():
    assert black_scholes("PUT",90,100,0,.05,.2) == 10
    assert black_scholes("CALL",100,100,1,.05,0) == pytest.approx(100-100*exp(-.05))
    for inputs in [("x",100,100,1,0,.2),("CALL",0,100,1,0,.2),
                   ("PUT",100,100,-1,0,.2),("PUT",100,100,1,0,float('nan'))]:
        with pytest.raises(ValueError):
            black_scholes(*inputs)
    with pytest.raises(ValueError, match="probability"):
        crr_price("PUT",100,100,1,1,.001,steps=1)
    with pytest.raises(ValueError):
        bsm_greeks("CALL",100,100,0,0,.2)
    for price in [-1, 100, 200]:
        with pytest.raises(ValueError):
            implied_volatility(price,"CALL",100,100,1,0)
    with pytest.raises(ValueError):
        implied_volatility(1,"CALL",100,100,0,0)


def test_option_right_is_one_enum_with_one_sign_convention():
    from cortexbacktest.options.contracts import OptionContract
    from cortexbacktest.options.pricing import OptionRight, option_right

    assert OptionRight.CALL == "CALL" and OptionRight.PUT == "PUT"  # stored words
    assert (OptionRight.CALL.sign, OptionRight.PUT.sign) == (1, -1)
    with pytest.raises(ValueError, match="CALL or PUT"):
        option_right("call")
    contract = OptionContract("SPY_P", "SPY", "PUT", 100.0, datetime(2026, 1, 16, tzinfo=timezone.utc))
    assert contract.right is OptionRight.PUT
    assert contract.intrinsic(90.0) == 10.0 and contract.intrinsic(110.0) == 0
