"""An account that stops being able to carry its positions is liquidated.

`MarginRequirementRule` already refuses orders an account cannot afford to open.
That covers the moment of entry and nothing after it: a position that was
affordable when it opened stops being affordable when the market moves against
it, without the account trading at all. Until now the engine let such an account
run a deficit indefinitely, so a strategy whose real outcome was "liquidated on
the third day" reported whatever the position did afterwards — including a
recovery it would not have been around to collect.

The two numbers that decide everything here are deliberately different:

- **equity** is what the account is worth — cash plus what the positions
  contributed, which for futures is only the unsettled variation margin;
- **the requirement** is measured on *notional*, the full contract value.

An account can be solvent on the first and short on the second. That is not an
edge case; it is the ordinary way a leveraged account is liquidated.
"""

from cortexbacktest import FillPrice
from datetime import datetime

import pytest

from cortexbacktest import (
    BacktestEngine,
    Bar,
    BrokerSimulator,
    CommissionScheme,
    Order,
    PercentCommission,
    Side,
    Strategy,
)

SYMBOL = "ES"
MULTIPLIER = 50.0


class BuyOnce(Strategy):
    def __init__(self, quantity: float = 1.0, side: Side = Side.BUY, symbol: str | None = None) -> None:
        self.done = False
        self.quantity = quantity
        self.side = side
        self.symbol = symbol

    def on_bars(self, bars):
        if self.done:
            return []
        self.done = True
        bar = bars[0] if self.symbol is None else next(b for b in bars if b.symbol == self.symbol)
        return [Order(bar.symbol, bar.timestamp, self.side, self.quantity)]


class BuyEachOnce(Strategy):
    """One order per symbol on the first bar; a negative quantity sells."""

    def __init__(self, quantities: dict[str, float]) -> None:
        self.quantities = quantities
        self.done = False

    def on_bars(self, bars):
        if self.done:
            return []
        self.done = True
        orders = []
        for bar in bars:
            signed = self.quantities.get(bar.symbol)
            if not signed:
                continue
            side = Side.BUY if signed > 0 else Side.SELL
            orders.append(Order(bar.symbol, bar.timestamp, side, abs(signed)))
        return orders


def _bars(prices: list[float], symbol: str = SYMBOL) -> list[Bar]:
    return [
        Bar(symbol, datetime(2024, 1, day + 1), price, price, price, price)
        for day, price in enumerate(prices)
    ]


def _interleave(*series: list[Bar]) -> list[Bar]:
    """Merge equal-length per-symbol bar lists into one time-ordered feed."""
    return [bar for group in zip(*series) for bar in group]


def _futures_broker(cash: float, **kwargs) -> BrokerSimulator:
    kwargs.setdefault("fill_price", FillPrice.CLOSE)
    return BrokerSimulator(
        initial_cash=cash,
        contract_multipliers={SYMBOL: MULTIPLIER},
        futures_symbols=[SYMBOL],
        **kwargs,
    )


# --- the rate has to be configured -----------------------------------------

def test_no_rate_means_no_liquidation():
    # The default stays what every existing backtest already assumes.
    broker = _futures_broker(20_000.0)

    BacktestEngine(trade_on_close=True, broker=broker).run(_bars([5000.0, 4000.0, 3000.0]), BuyOnce())

    assert broker.margin_calls == []
    assert broker.state.positions[SYMBOL].quantity == pytest.approx(1.0)


def test_a_non_positive_rate_is_rejected():
    with pytest.raises(ValueError, match="maintenance_margin must be positive"):
        BrokerSimulator(fill_price=FillPrice.CLOSE, initial_cash=10_000.0, maintenance_margin=0.0)


def test_the_rate_can_come_from_the_commission_scheme():
    # CommissionScheme.maintenance_margin was declared, validated, and read by
    # nothing until now.
    scheme = CommissionScheme(
        commission_model=PercentCommission(0.0),
        initial_margin=0.05,
        maintenance_margin=0.04,
    )
    broker = BrokerSimulator(fill_price=FillPrice.CLOSE, initial_cash=10_000.0, commission_scheme=scheme)

    assert broker.maintenance_margin == pytest.approx(0.04)


def test_an_explicit_rate_wins_over_the_scheme():
    scheme = CommissionScheme(
        commission_model=PercentCommission(0.0),
        maintenance_margin=0.04,
    )
    broker = BrokerSimulator(fill_price=FillPrice.CLOSE,
        initial_cash=10_000.0, commission_scheme=scheme, maintenance_margin=0.10
    )

    assert broker.maintenance_margin == pytest.approx(0.10)


# --- a comfortable account is left alone -----------------------------------

def test_an_account_within_its_requirement_is_not_touched():
    # $50,000 against $250,000 of notional at a 4% requirement: $10,000 required
    # against $50,000 of equity, with room to spare.
    broker = _futures_broker(50_000.0, maintenance_margin=0.04)

    BacktestEngine(trade_on_close=True, broker=broker).run(_bars([5000.0, 5010.0, 5020.0]), BuyOnce())

    assert broker.margin_calls == []
    assert broker.state.positions[SYMBOL].quantity == pytest.approx(1.0)


def test_a_loss_that_stays_inside_the_requirement_is_not_a_call():
    broker = _futures_broker(50_000.0, maintenance_margin=0.04)

    BacktestEngine(trade_on_close=True, broker=broker).run(_bars([5000.0, 4900.0, 4900.0]), BuyOnce())

    # Down $5,000 on the move, still $45,000 of equity against $9,800 required.
    assert broker.margin_calls == []


# --- the breach -------------------------------------------------------------

def test_a_move_through_the_requirement_liquidates():
    # 4 contracts at 5000 x 50 = $1,000,000 notional; at 20% that is $200,000
    # required against $220,000 of equity. A 100-point drop costs $20,000 and
    # leaves $200,000 of equity against $196,000 required -- still fine. A
    # 400-point drop costs $80,000: $140,000 of equity against $184,000.
    broker = _futures_broker(220_000.0, maintenance_margin=0.20)

    BacktestEngine(trade_on_close=True, broker=broker).run(_bars([5000.0, 4600.0, 4600.0]), BuyOnce(quantity=4.0))

    assert len(broker.margin_calls) == 1
    call = broker.margin_calls[0]
    assert call.equity == pytest.approx(140_000.0)
    assert call.requirement == pytest.approx(184_000.0)
    assert call.shortfall == pytest.approx(44_000.0)
    assert call.resolved


def test_liquidation_closes_only_what_the_shortfall_needs():
    broker = _futures_broker(220_000.0, maintenance_margin=0.20)

    BacktestEngine(trade_on_close=True, broker=broker).run(_bars([5000.0, 4600.0, 4600.0]), BuyOnce(quantity=4.0))

    # Each contract released at 4600 frees 4600 x 50 x 0.20 = $46,000 of
    # requirement, so a $44,000 shortfall takes 44,000/46,000 of one contract.
    liquidation = broker.margin_calls[0].liquidations[0]
    assert liquidation.quantity == pytest.approx(-44_000.0 / 46_000.0)
    assert liquidation.price == pytest.approx(4600.0)
    assert broker.state.positions[SYMBOL].quantity == pytest.approx(4.0 - 44_000.0 / 46_000.0)


def test_the_account_ends_the_call_inside_its_requirement():
    broker = _futures_broker(220_000.0, maintenance_margin=0.20)

    BacktestEngine(trade_on_close=True, broker=broker).run(_bars([5000.0, 4600.0, 4600.0]), BuyOnce(quantity=4.0))

    state = broker.state
    assert state.maintenance_requirement(0.20) <= state.equity() + 1e-6


def test_liquidating_does_not_change_equity_for_futures():
    # Closing a futures position realises unsettled P&L into cash; it moves
    # nothing between the account and the market. Without commission the
    # liquidation is equity-neutral, which is what makes it a forced *exit*
    # rather than a forced loss.
    broker = _futures_broker(220_000.0, maintenance_margin=0.20)

    BacktestEngine(trade_on_close=True, broker=broker).run(_bars([5000.0, 4600.0, 4600.0]), BuyOnce(quantity=4.0))

    assert broker.state.equity() == pytest.approx(140_000.0)


def test_the_liquidation_is_a_fill_like_any_other():
    broker = _futures_broker(220_000.0, maintenance_margin=0.20)

    result = BacktestEngine(trade_on_close=True, broker=broker).run(
        _bars([5000.0, 4600.0, 4600.0]), BuyOnce(quantity=4.0)
    )

    # Two fills: the strategy's entry and the forced exit. A liquidation that
    # bypassed the fill record would break PnL reconciliation.
    assert len(result.fills) == 2
    forced = result.fills[1]
    assert forced.side == Side.SELL
    assert forced.price == pytest.approx(4600.0)


def test_commission_is_charged_on_the_forced_exit():
    broker = _futures_broker(220_000.0, maintenance_margin=0.20, commission_bps=1.0)

    BacktestEngine(trade_on_close=True, broker=broker).run(_bars([5000.0, 4600.0, 4600.0]), BuyOnce(quantity=4.0))

    liquidation = broker.margin_calls[0].liquidations[0]
    notional = abs(liquidation.quantity) * 4600.0 * MULTIPLIER
    assert liquidation.commission == pytest.approx(notional * 0.0001)
    assert liquidation.commission > 0


# --- shorts and equities ----------------------------------------------------

def test_a_short_is_covered_rather_than_sold():
    broker = _futures_broker(220_000.0, maintenance_margin=0.20)

    BacktestEngine(trade_on_close=True, broker=broker).run(
        _bars([5000.0, 5400.0, 5400.0]), BuyOnce(quantity=4.0, side=Side.SELL)
    )

    liquidation = broker.margin_calls[0].liquidations[0]
    assert liquidation.quantity > 0
    assert broker.state.positions[SYMBOL].quantity == pytest.approx(-4.0 + liquidation.quantity)


def test_a_cash_equity_account_is_liquidated_on_the_same_rule():
    # Nothing here is futures-specific. A long cash position cannot breach --
    # with no borrowing its notional *is* its equity, and the requirement is a
    # fraction of that. A short can: the proceeds sit in cash while the
    # liability grows with the price. Short 10 at $100 on $5,000 leaves $6,000
    # of cash against a $1,000 liability, so $5,000 of equity against $300
    # required at 30%. At $150 that is $4,500 of equity against $450 -- still
    # inside. At $1,400 it is $6,000 - $14,000 = -$8,000 against $4,200.
    broker = BrokerSimulator(fill_price=FillPrice.CLOSE, initial_cash=5_000.0, maintenance_margin=0.30)

    BacktestEngine(trade_on_close=True, broker=broker).run(
        _bars([100.0, 300.0, 300.0], symbol="AAPL"), BuyOnce(quantity=10.0, side=Side.SELL)
    )

    # At $300 the short is worth -$3,000 against $6,000 of cash: $3,000 of
    # equity, $900 required. Comfortable, so the position survives.
    assert broker.margin_calls == []
    assert broker.state.positions["AAPL"].quantity == pytest.approx(-10.0)


def test_a_short_equity_position_breaches_when_the_price_runs():
    broker = BrokerSimulator(fill_price=FillPrice.CLOSE, initial_cash=5_000.0, maintenance_margin=0.30)

    BacktestEngine(trade_on_close=True, broker=broker).run(
        _bars([100.0, 1_400.0, 1_400.0], symbol="AAPL"), BuyOnce(quantity=10.0, side=Side.SELL)
    )

    call = broker.margin_calls[0]
    assert call.equity == pytest.approx(-8_000.0)
    assert call.requirement == pytest.approx(4_200.0)
    # Covering a short at a loss spends cash, so equity falls no further but the
    # liability stops growing. Nothing left to sell means the call stands.
    assert not call.resolved
    assert broker.state.positions == {}


def test_the_largest_position_is_liquidated_first():
    small = _bars([100.0, 120.0, 120.0], symbol="SMALL")
    large = _bars([1_000.0, 1_200.0, 1_200.0], symbol="LARGE")
    broker = BrokerSimulator(fill_price=FillPrice.CLOSE, initial_cash=5_000.0, maintenance_margin=0.30)

    BacktestEngine(trade_on_close=True, broker=broker).run(
        _interleave(small, large),
        BuyEachOnce({"SMALL": -10.0, "LARGE": -10.0}),
    )

    # Both shorted at the open: cash 5,000 + 1,000 + 10,000 = 16,000 against
    # 11,000 of liability. A 20% run leaves 16,000 - 13,200 = 2,800 of equity
    # against 0.30 x 13,200 = 3,960 required, a shortfall of 1,160. Covering
    # LARGE frees 1,200 x 0.30 = 360 per share, so 3.22 shares clear it and
    # SMALL is never touched.
    call = broker.margin_calls[0]
    assert [liquidation.symbol for liquidation in call.liquidations] == ["LARGE"]
    assert call.liquidations[0].quantity == pytest.approx(1_160.0 / 360.0)
    assert broker.state.positions["SMALL"].quantity == pytest.approx(-10.0)


# --- the account that cannot be saved ---------------------------------------

def test_an_unresolvable_call_is_reported_as_unresolved():
    # Equity below zero: liquidating everything still leaves the account short,
    # because the deficit is cash, not exposure. Reporting `resolved=True` here
    # would hide the only outcome that matters.
    broker = _futures_broker(220_000.0, maintenance_margin=0.20)

    BacktestEngine(trade_on_close=True, broker=broker).run(_bars([5000.0, 3000.0, 3000.0]), BuyOnce(quantity=4.0))

    call = broker.margin_calls[0]
    assert call.equity < 0
    assert not call.resolved
    assert SYMBOL not in broker.state.positions


def test_everything_is_closed_when_nothing_less_will_do():
    broker = _futures_broker(220_000.0, maintenance_margin=0.20)

    BacktestEngine(trade_on_close=True, broker=broker).run(_bars([5000.0, 3000.0, 3000.0]), BuyOnce(quantity=4.0))

    assert broker.state.positions == {}
    assert broker.margin_calls[0].liquidations[0].quantity == pytest.approx(-4.0)


# --- interaction with the rest of the session ------------------------------

def test_resting_orders_on_a_liquidated_symbol_are_cancelled():
    class BuyThenRestALimit(Strategy):
        def __init__(self):
            self.index = -1

        def on_bars(self, bars):
            self.index += 1
            bar = bars[0]
            if self.index == 0:
                return [Order(bar.symbol, bar.timestamp, Side.BUY, 4.0)]
            if self.index == 1:
                return [
                    Order(bar.symbol, bar.timestamp, Side.BUY, 1.0, limit_price=1.0)
                ]
            return []

    broker = _futures_broker(220_000.0, maintenance_margin=0.20)

    BacktestEngine(trade_on_close=True, broker=broker).run(
        _bars([5000.0, 4600.0, 4600.0, 4600.0]), BuyThenRestALimit()
    )

    # The limit was placed on the second bar and the call fires at the third
    # session's open, so the order is on the book when the liquidation runs.
    assert broker.open_orders == []


def test_the_call_is_measured_after_variation_margin_settles():
    # Order matters: yesterday's futures loss is cash before the requirement is
    # tested, so the same account is tested against the same equity whichever
    # way the P&L happens to be booked at that instant.
    broker = _futures_broker(220_000.0, maintenance_margin=0.20)

    BacktestEngine(trade_on_close=True, broker=broker).run(_bars([5000.0, 4600.0, 4600.0]), BuyOnce(quantity=4.0))

    call = broker.margin_calls[0]
    # Equity at the call equals cash after settlement, since the settled basis
    # leaves no unsettled variation margin to count.
    assert call.equity == pytest.approx(140_000.0)


def test_one_call_per_session_at_most():
    broker = _futures_broker(220_000.0, maintenance_margin=0.20)

    BacktestEngine(trade_on_close=True, broker=broker).run(
        _bars([5000.0, 4600.0, 4600.0, 4600.0, 4600.0]), BuyOnce(quantity=4.0)
    )

    # The first call brings the account inside its requirement and the price
    # then stops moving, so no further session can breach.
    assert len(broker.margin_calls) == 1


# --- reachable from the stable API -----------------------------------------

def test_the_stable_api_can_configure_both_futures_and_maintenance_margin():
    # An option the public config cannot express is an option most callers do
    # not have. BrokerConfig carried neither of these until now.
    from cortexbacktest import BrokerConfig

    config = BrokerConfig(
        initial_cash=220_000.0,
        contract_multipliers={SYMBOL: MULTIPLIER},
        futures_symbols=(SYMBOL,),
        maintenance_margin=0.20,
    )
    broker = config.create_broker()

    assert broker.maintenance_margin == pytest.approx(0.20)
    assert broker.state.is_futures(SYMBOL)
