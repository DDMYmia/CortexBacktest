"""Cash dividends settle on the pay date, net of withholding tax.

A dividend has two dates and they are weeks apart. Entitlement is fixed at the
ex-date, on the shares held that morning; the cash arrives on the pay date. Until
now the engine collapsed the two, and a backtest could spend a dividend the
instant it went ex — money the account would not actually hold for another
fortnight. Buying power is the thing that reads wrong: equity is the same either
way, because the receivable is worth what the cash will be.

Withholding is the other half. A received dividend is taxed at the payer's
jurisdiction; a short pays the *gross*, because the borrower owes the lender the
whole amount and no tax authority is involved in that transfer. Applying the
rate to both sides would make being short across an ex-date profitable, which is
the kind of error a strategy search finds and exploits.
"""

from cortexbacktest import FillPrice
from datetime import date, datetime

import pytest

from cortexbacktest import (
    BacktestEngine,
    Bar,
    BrokerSimulator,
    CorporateAction,
    CorporateActionType,
    Order,
    Side,
    Strategy,
)

SYMBOL = "AAA"
EX_DATE = date(2024, 1, 3)
PAY_DATE = date(2024, 1, 5)


class BuyFirstBar(Strategy):
    def __init__(self, quantity: float = 100.0, side: Side = Side.BUY) -> None:
        self.done = False
        self.quantity = quantity
        self.side = side

    def on_bars(self, bars: list[Bar]) -> list[Order]:
        if self.done:
            return []
        self.done = True
        return [Order(bars[0].symbol, bars[0].timestamp, self.side, self.quantity)]


class BuyThenSellOn(Strategy):
    """Buy on the first bar, flatten on the bar at `sell_index`."""

    def __init__(self, sell_index: int, quantity: float = 100.0) -> None:
        self.sell_index = sell_index
        self.quantity = quantity
        self.index = -1

    def on_bars(self, bars: list[Bar]) -> list[Order]:
        self.index += 1
        bar = bars[0]
        if self.index == 0:
            return [Order(bar.symbol, bar.timestamp, Side.BUY, self.quantity)]
        if self.index == self.sell_index:
            return [Order(bar.symbol, bar.timestamp, Side.SELL, self.quantity)]
        return []


def _bars(count: int = 6, price: float = 100.0) -> list[Bar]:
    return [
        Bar(SYMBOL, datetime(2024, 1, day + 1), price, price, price, price)
        for day in range(count)
    ]


def _dividend(**kwargs) -> CorporateAction:
    return CorporateAction(
        SYMBOL, EX_DATE, CorporateActionType.CASH_DIVIDEND, amount=kwargs.pop("amount", 2.0), **kwargs
    )


def _run(actions, strategy=None, **broker_kwargs):
    broker_kwargs.setdefault("fill_price", FillPrice.CLOSE)
    broker = BrokerSimulator(initial_cash=100_000.0, corporate_actions=actions, **broker_kwargs)
    result = BacktestEngine(trade_on_close=True, broker=broker).run(_bars(), strategy or BuyFirstBar())
    return broker, result


def _column(result, name: str) -> list[float]:
    return result.performance.equity_curve()[name].to_list()


# --- the default is unchanged ----------------------------------------------

def test_a_dividend_without_a_pay_date_still_settles_at_the_ex_date():
    # Every backtest written before pay_date existed assumes this.
    broker, result = _run([_dividend()])

    # 100 shares at 2.00 lands on the ex-date session, the third bar.
    assert _column(result, "cash")[2] - _column(result, "cash")[1] == pytest.approx(200.0)
    assert broker.state.pending_cash == pytest.approx(0.0)


# --- pay-date settlement ---------------------------------------------------

def test_the_cash_arrives_on_the_pay_date_not_the_ex_date():
    broker, result = _run([_dividend(pay_date=PAY_DATE)])

    cash = _column(result, "cash")
    # Nothing on the ex-date (index 2); the full 200 on the pay date (index 4).
    assert cash[2] == pytest.approx(cash[1])
    assert cash[4] - cash[3] == pytest.approx(200.0)


def test_the_receivable_is_held_as_pending_cash_in_between():
    broker, result = _run([_dividend(pay_date=PAY_DATE)])

    pending = _column(result, "pending_cash")
    assert pending[2] == pytest.approx(200.0)
    assert pending[3] == pytest.approx(200.0)
    assert pending[4] == pytest.approx(0.0)


def test_equity_is_identical_whichever_date_the_cash_lands_on():
    # The receivable is worth what the cash will be, so the only thing pay-date
    # settlement changes is when the account can spend it.
    _, immediate = _run([_dividend()])
    _, delayed = _run([_dividend(pay_date=PAY_DATE)])

    assert _column(immediate, "equity") == pytest.approx(_column(delayed, "equity"))


def test_a_pay_date_equal_to_the_ex_date_settles_immediately():
    broker, result = _run([_dividend(pay_date=EX_DATE)])

    assert _column(result, "cash")[2] - _column(result, "cash")[1] == pytest.approx(200.0)
    assert broker.state.pending_cash == pytest.approx(0.0)


def test_a_pay_date_with_no_session_settles_at_the_next_one():
    # Omit the pay date from this synthetic feed; settlement must wait until
    # the next observed bar rather than disappear in the gap.
    broker = BrokerSimulator(fill_price=FillPrice.CLOSE, initial_cash=100_000.0, corporate_actions=[_dividend(pay_date=date(2024, 1, 6))])
    bars = [bar for bar in _bars(7) if bar.timestamp.day != 6]
    result = BacktestEngine(trade_on_close=True, broker=broker).run(bars, BuyFirstBar())

    assert broker.state.pending_cash == pytest.approx(0.0)
    assert _column(result, "cash")[-1] - _column(result, "cash")[3] == pytest.approx(200.0)


def test_selling_before_the_pay_date_keeps_the_dividend():
    # Entitlement is fixed at the ex-date. Selling on the fourth bar, one day
    # after going ex, does not hand the dividend back.
    broker, result = _run([_dividend(pay_date=PAY_DATE)], BuyThenSellOn(sell_index=3))

    assert broker.state.positions == {}
    assert _column(result, "cash")[4] - _column(result, "cash")[3] == pytest.approx(200.0)


# --- withholding tax -------------------------------------------------------

def test_the_broker_default_rate_is_withheld_from_a_received_dividend():
    broker, result = _run([_dividend()], dividend_withholding_tax=0.15)

    assert _column(result, "cash")[2] - _column(result, "cash")[1] == pytest.approx(170.0)


def test_an_action_rate_overrides_the_broker_default():
    # The rate follows the payer's jurisdiction, so it belongs to the action
    # when the two disagree.
    broker, result = _run([_dividend(withholding_tax=0.30)], dividend_withholding_tax=0.15)

    assert _column(result, "cash")[2] - _column(result, "cash")[1] == pytest.approx(140.0)


def test_a_zero_rate_on_the_action_overrides_a_non_zero_default():
    # `None` means "defer to the broker"; 0.0 means "this one is untaxed", and
    # collapsing the two would make a tax-exempt issuer unexpressible.
    broker, result = _run([_dividend(withholding_tax=0.0)], dividend_withholding_tax=0.15)

    assert _column(result, "cash")[2] - _column(result, "cash")[1] == pytest.approx(200.0)


def test_a_short_pays_the_gross_dividend():
    # The borrower owes the lender the whole amount. Taxing this side too would
    # pay the short 15% of every dividend it was short across.
    broker, result = _run(
        [_dividend()], BuyFirstBar(side=Side.SELL), dividend_withholding_tax=0.15
    )

    assert _column(result, "cash")[2] - _column(result, "cash")[1] == pytest.approx(-200.0)


def test_tax_is_withheld_from_a_delayed_dividend_too():
    broker, result = _run([_dividend(pay_date=PAY_DATE)], dividend_withholding_tax=0.15)

    assert _column(result, "pending_cash")[2] == pytest.approx(170.0)
    assert _column(result, "cash")[4] - _column(result, "cash")[3] == pytest.approx(170.0)


# --- what the event reports ------------------------------------------------

def test_the_event_reports_the_net_amount_and_the_tax():
    broker, _ = _run([_dividend()], dividend_withholding_tax=0.15)

    event = broker.corporate_action_events[0]
    assert event.cash_amount == pytest.approx(170.0)
    assert event.tax_amount == pytest.approx(30.0)


def test_the_event_reports_when_the_cash_lands():
    broker, _ = _run([_dividend(pay_date=PAY_DATE)])

    assert broker.corporate_action_events[0].settles_on == PAY_DATE


def test_an_undated_dividend_reports_settling_on_its_ex_date():
    broker, _ = _run([_dividend()])

    assert broker.corporate_action_events[0].settles_on == EX_DATE


def test_a_short_reports_no_tax():
    broker, _ = _run([_dividend()], BuyFirstBar(side=Side.SELL), dividend_withholding_tax=0.15)

    event = broker.corporate_action_events[0]
    assert event.tax_amount == pytest.approx(0.0)
    assert event.cash_amount == pytest.approx(-200.0)


# --- rejected inputs -------------------------------------------------------

def test_a_pay_date_before_the_ex_date_is_rejected():
    with pytest.raises(ValueError, match="pay_date must not precede ex_date"):
        _dividend(pay_date=date(2024, 1, 1))


def test_a_withholding_rate_outside_zero_to_one_is_rejected():
    with pytest.raises(ValueError, match=r"withholding_tax must be in \[0, 1\)"):
        _dividend(withholding_tax=1.0)
    with pytest.raises(ValueError, match=r"withholding_tax must be in \[0, 1\)"):
        _dividend(withholding_tax=-0.1)


def test_the_broker_rejects_a_default_rate_outside_zero_to_one():
    with pytest.raises(ValueError, match=r"dividend_withholding_tax must be in \[0, 1\)"):
        BrokerSimulator(fill_price=FillPrice.CLOSE, initial_cash=1_000.0, dividend_withholding_tax=1.0)


@pytest.mark.parametrize(
    "action_type,extra",
    [
        (CorporateActionType.SPLIT, {"ratio": 2.0}),
        (CorporateActionType.STOCK_DIVIDEND, {"ratio": 0.1}),
        (CorporateActionType.DELISTING, {"amount": 1.0}),
    ],
)
def test_other_action_types_reject_the_dividend_only_fields(action_type, extra):
    # Silently ignoring them is how a run reports a settlement date it never
    # applied.
    with pytest.raises(ValueError, match="must not carry pay_date"):
        CorporateAction(SYMBOL, EX_DATE, action_type, pay_date=PAY_DATE, **extra)
    with pytest.raises(ValueError, match="must not carry withholding_tax"):
        CorporateAction(SYMBOL, EX_DATE, action_type, withholding_tax=0.1, **extra)


def test_the_stable_api_can_set_the_default_withholding_rate():
    from cortexbacktest import BrokerConfig

    broker = BrokerConfig(initial_cash=100_000.0, dividend_withholding_tax=0.15).create_broker()

    assert broker.dividend_withholding_tax == pytest.approx(0.15)
