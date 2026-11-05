"""PnL reconciliation through corporate actions.

Equity moves with every split, dividend, delisting and merger the broker applies,
so a reconciliation built from fills alone misses them: a split re-pairs lots at
pre-split prices against post-split quantities, and dividends never appear. Each
test runs a real backtest through one action and requires the fill-level PnL to
reconcile to final equity exactly.
"""

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


class Script(Strategy):
    """Submit the listed orders on the given bar index: {index: [(symbol, side, quantity)]}."""

    def __init__(self, plan: dict[int, list[tuple[str, Side, float]]]) -> None:
        self.plan = plan
        self.index = -1
        self.seen: set = set()

    def on_bars(self, bars: list[Bar]) -> list[Order]:
        stamp = bars[0].timestamp
        if stamp not in self.seen:
            self.seen.add(stamp)
            self.index += 1
        else:
            return []
        return [Order(symbol, stamp, side, quantity) for symbol, side, quantity in self.plan.get(self.index, [])]


def _bars(prices: dict[str, list[float]]) -> list[Bar]:
    return sorted(
        (Bar(symbol, datetime(2024, 1, day + 1), p, p * 1.01, p * 0.99, p)
         for symbol, series in prices.items() for day, p in enumerate(series)),
        key=lambda bar: (bar.timestamp, bar.symbol),
    )


def _run(prices, plan, actions):
    broker = BrokerSimulator(initial_cash=100_000.0, commission_bps=5.0, corporate_actions=actions)
    result = BacktestEngine(broker=broker).run(_bars(prices), Script(plan))
    return result, result.trades.pnl_summary()


def test_a_split_rescales_open_lots_so_the_round_trip_is_priced_correctly():
    # Bought 100 at ~100, split 2:1, sold 200 at ~55: a 10% gain, not a loss on half the lot.
    prices = {"AAA": [100.0, 100.0, 50.0, 55.0, 55.0, 55.0]}
    actions = [CorporateAction("AAA", date(2024, 1, 3), CorporateActionType.SPLIT, ratio=2.0)]
    result, summary = _run(prices, {0: [("AAA", Side.BUY, 100.0)], 3: [("AAA", Side.SELL, 200.0)]}, actions)
    assert summary["reconciliation_error"] == pytest.approx(0.0, abs=1e-6)
    trades = result.trades.frame()
    assert trades["quantity"].sum() == pytest.approx(200.0)
    assert trades["gross_pnl"].sum() == pytest.approx(1_000.0)


def test_cash_dividends_are_income_held_or_short():
    prices = {"AAA": [100.0] * 5, "BBB": [50.0] * 5}
    actions = [
        CorporateAction("AAA", date(2024, 1, 3), CorporateActionType.CASH_DIVIDEND, amount=1.0, pay_date=date(2024, 1, 20)),
        CorporateAction("BBB", date(2024, 1, 4), CorporateActionType.CASH_DIVIDEND, amount=0.5),
    ]
    plan = {0: [("AAA", Side.BUY, 100.0), ("BBB", Side.SELL, 40.0)]}
    _, summary = _run(prices, plan, actions)
    assert summary["dividends"] == pytest.approx(100.0 - 20.0)
    assert summary["reconciliation_error"] == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("action", [
    CorporateAction("AAA", date(2024, 1, 3), CorporateActionType.DELISTING, amount=30.0),
    CorporateAction("AAA", date(2024, 1, 3), CorporateActionType.STOCK_DIVIDEND, ratio=0.1),
    CorporateAction("AAA", date(2024, 1, 3), CorporateActionType.SYMBOL_CHANGE, new_symbol="CCC"),
    CorporateAction("AAA", date(2024, 1, 3), CorporateActionType.MERGER, new_symbol="CCC", ratio=0.5, amount=10.0),
    CorporateAction("AAA", date(2024, 1, 3), CorporateActionType.MERGER, new_symbol="CCC", ratio=0.0, amount=120.0),
], ids=["delisting", "stock_dividend", "symbol_change", "stock_and_cash_merger", "cash_merger"])
@pytest.mark.parametrize("side", [Side.BUY, Side.SELL], ids=["long", "short"])
def test_every_action_reconciles_long_and_short(action, side):
    prices = {"AAA": [100.0, 100.0], "CCC": [180.0, 180.0, 185.0, 190.0, 190.0]}
    prices["AAA"] += [101.0, 102.0, 102.0] if action.action_type in (CorporateActionType.SPLIT, CorporateActionType.STOCK_DIVIDEND) else []
    _, summary = _run(prices, {0: [("AAA", side, 100.0)]}, [action])
    assert summary["reconciliation_error"] == pytest.approx(0.0, abs=1e-6)


def test_without_records_of_the_actions_the_gap_is_the_actions_themselves():
    from cortexbacktest import pnl_reconciliation_summary

    prices = {"AAA": [100.0, 100.0, 50.0, 55.0, 55.0, 55.0]}
    actions = [CorporateAction("AAA", date(2024, 1, 3), CorporateActionType.SPLIT, ratio=2.0)]
    result, _ = _run(prices, {0: [("AAA", Side.BUY, 100.0)], 3: [("AAA", Side.SELL, 200.0)]}, actions)
    blind = pnl_reconciliation_summary(result.fills, result.positions, initial_equity=result.initial_equity,
                                       final_equity=result.performance.equity_curve()["equity"][-1])
    assert abs(blind["reconciliation_error"]) > 100.0
