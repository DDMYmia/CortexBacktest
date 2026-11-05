from datetime import datetime, timedelta, timezone

import pytest

from cortexbacktest.options import (
    OptionBacktestConfig, OptionContract, OptionMarketFrame, OptionQuote, ResearchOptionCosts,
    run_option_backtest, run_option_scenarios,
)

T = datetime(2026, 1, 5, 20, tzinfo=timezone.utc)
CONTRACT = OptionContract("C", "S", "CALL", 100, T + timedelta(days=30))


def quote(bid=2.0, ask=3.0):
    return OptionQuote("C", T, T, bid, ask, bid_size=100, ask_size=100)


def price(costs, quantity, *, is_option=True, bid=2.0, ask=3.0, multiplier=100):
    return costs.execution("C", quantity, quote(bid, ask), multiplier=multiplier,
                           is_option=is_option)


def test_defaults_reproduce_the_engines_previous_behaviour():
    """The plugin must not move an existing result: full half-spread, 0.65 a contract."""
    costs = ResearchOptionCosts()
    assert price(costs, 1) == (3.0, 0.65)          # buy lifts the offer
    assert price(costs, -2) == (2.0, 1.30)         # sell hits the bid
    assert price(ResearchOptionCosts(spread_fraction=0.0), 1) == (2.5, 0.65)
    assert price(ResearchOptionCosts(spread_fraction=0.5), 1) == (2.75, 0.65)


def test_slippage_is_charged_beyond_the_touch_and_is_signed():
    costs = ResearchOptionCosts(option_slippage_ticks=3, tick_size=0.01)
    assert price(costs, 1)[0] == pytest.approx(3.03)
    assert price(costs, -1)[0] == pytest.approx(1.97)
    # A stock fill slips in basis points of its own price, not in option ticks.
    stock = ResearchOptionCosts(stock_slippage_bps=10)
    assert price(stock, 1, is_option=False)[0] == pytest.approx(3.0 + 2.5 * 0.001)
    assert price(stock, 1)[0] == pytest.approx(3.0)


def test_impact_follows_the_square_root_of_size():
    costs = ResearchOptionCosts(impact_coefficient=1.0, impact_reference=100)
    half_spread = 0.5
    assert price(costs, 100)[0] == pytest.approx(3.0 + half_spread)      # one reference size
    assert price(costs, 400)[0] == pytest.approx(3.0 + 2 * half_spread)  # four times, twice the cost
    assert price(costs, 25)[0] == pytest.approx(3.0 + 0.5 * half_spread)
    assert price(ResearchOptionCosts(), 400)[0] == pytest.approx(3.0)    # off by default


def test_commission_supports_per_contract_premium_and_ticket_minimums():
    per_contract = ResearchOptionCosts(option_commission=0.5)
    assert price(per_contract, 4)[1] == pytest.approx(2.0)

    on_premium = ResearchOptionCosts(option_commission=0.0, option_commission_bps=10)
    # 1 contract x 100 multiplier x 3.00 premium = 300 notional, 10bps = 0.30
    assert price(on_premium, 1)[1] == pytest.approx(0.30)

    ticket = ResearchOptionCosts(option_commission=0.5, option_minimum_ticket=5.0)
    assert price(ticket, 1)[1] == pytest.approx(5.0)
    assert price(ticket, 20)[1] == pytest.approx(10.0)

    stock = ResearchOptionCosts(stock_commission_bps=5)
    assert price(stock, 100, is_option=False)[1] == pytest.approx(100 * 3.0 * 0.0005)


def test_a_fill_price_cannot_go_negative():
    """Slippage may exceed a near-worthless premium; it must not pay the seller."""
    costs = ResearchOptionCosts(option_slippage_ticks=100)
    assert price(costs, -1, bid=0.01, ask=0.05)[0] == 0.0


def test_invalid_components_are_rejected():
    for kwargs in ({"spread_fraction": 1.5}, {"option_commission": -1},
                   {"tick_size": float("nan")}, {"impact_reference": 0}):
        with pytest.raises(ValueError):
            ResearchOptionCosts(**kwargs)
    with pytest.raises(ValueError, match="nonzero quantity"):
        price(ResearchOptionCosts(), 0)


def _run(config, targets):
    frames = [OptionMarketFrame(T + timedelta(days=day), {
        "C": OptionQuote("C", T + timedelta(days=day), T + timedelta(days=day), 2.0, 3.0,
                         bid_size=100, ask_size=100),
        "S": OptionQuote("S", T + timedelta(days=day), T + timedelta(days=day), 99.9, 100.1,
                         bid_size=10_000, ask_size=10_000)}) for day in range(3)]
    return run_option_backtest([CONTRACT], frames,
                               lambda frame, account: targets.get(frame.timestamp, {}),
                               config=config)


def test_the_plugin_and_the_flat_fields_cannot_both_be_set():
    with pytest.raises(ValueError, match="costs plugin"):
        OptionBacktestConfig(costs=ResearchOptionCosts(), option_fee=1.0)
    with pytest.raises(ValueError, match="costs plugin"):
        OptionBacktestConfig(costs=ResearchOptionCosts(), spread_fraction=0.5)


def test_the_run_charges_and_records_the_plugin():
    targets = {T: {"C": 2}}
    plain = _run(OptionBacktestConfig(), targets)
    slipped = _run(OptionBacktestConfig(
        costs=ResearchOptionCosts(option_slippage_ticks=5, option_commission=0.65)), targets)

    assert plain.fills["price"].to_list() == [3.0]
    assert slipped.fills["price"].to_list() == [3.05]
    # 2 contracts x 100 multiplier x 0.05 of extra slippage.
    assert slipped.attribution["option_spread"] - plain.attribution["option_spread"] == (
        pytest.approx(10.0))
    assert slipped.summary()["net_pnl"] < plain.summary()["net_pnl"]
    assert slipped.manifest["cost_model"] == "ResearchOptionCosts"
    assert slipped.manifest["cost_parameters"]["option_slippage_ticks"] == 5
    identity = slipped.attribution
    assert identity["net_pnl"] == pytest.approx(
        identity["option_market_pnl"] + identity["stock_market_pnl"] - identity["option_spread"]
        - identity["stock_spread"] - identity["option_fees"] - identity["stock_fees"]
        + identity["financing"] + identity["dividends"], abs=1e-8)


def test_a_spread_sweep_varies_the_plugin_not_a_dead_config_field():
    """With a plugin, replacing the config's own spread_fraction would compare
    five identical scenarios — the sweep has to reach inside the plugin."""
    targets = {T: {"C": 2}}
    frames = [OptionMarketFrame(T + timedelta(days=day), {
        "C": OptionQuote("C", T + timedelta(days=day), T + timedelta(days=day), 2.0, 3.0,
                         bid_size=100, ask_size=100),
        "S": OptionQuote("S", T + timedelta(days=day), T + timedelta(days=day), 99.9, 100.1,
                         bid_size=10_000, ask_size=10_000)}) for day in range(3)]
    config = OptionBacktestConfig(costs=ResearchOptionCosts(option_slippage_ticks=1))
    results = run_option_scenarios([CONTRACT], frames,
                                   lambda: (lambda frame, account: targets.get(frame.timestamp, {})),
                                   config=config, spread_fractions=(0.0, 0.5, 1.0))
    prices = [results[fraction].fills["price"][0] for fraction in (0.0, 0.5, 1.0)]
    assert prices == [pytest.approx(2.51), pytest.approx(2.76), pytest.approx(3.01)]

    class Opaque:
        def execution(self, symbol, quantity, quote, *, multiplier, is_option):
            return quote.mid, 0.0

    with pytest.raises(ValueError, match="spread_fraction field"):
        run_option_scenarios([CONTRACT], frames, lambda: (lambda f, a: {}),
                             config=OptionBacktestConfig(costs=Opaque()))
