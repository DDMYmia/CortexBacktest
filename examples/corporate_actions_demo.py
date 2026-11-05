"""Apply a split and a cash dividend to an event backtest.

Corporate actions run on unadjusted prices: the raw quote drops on the ex-date
and the action restores what the holder actually owns. Equity should therefore
step through both events smoothly.
"""

from __future__ import annotations

from datetime import date, datetime
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cortexbacktest import (  # noqa: E402
    BacktestEngine,
    Bar,
    BrokerSimulator,
    CorporateAction,
    CorporateActionType,
    Order,
    Side,
    Strategy,
)


class BuyAndHold(Strategy):
    def __init__(self) -> None:
        self.bought = False

    def on_bars(self, bars: list[Bar]) -> list[Order]:
        if self.bought:
            return []
        self.bought = True
        return [Order(bar.symbol, bar.timestamp, Side.BUY, 100) for bar in bars]


def main() -> int:
    # Unadjusted quotes: 100, 100, then 2-for-1 split (50), then a $1 dividend (49).
    prices = [100.0, 100.0, 50.0, 49.0, 49.0]
    bars = [
        Bar("AAA", datetime(2024, 1, day + 1), price, price, price, price)
        for day, price in enumerate(prices)
    ]
    actions = [
        CorporateAction("AAA", date(2024, 1, 3), CorporateActionType.SPLIT, ratio=2.0),
        CorporateAction("AAA", date(2024, 1, 4), CorporateActionType.CASH_DIVIDEND, amount=1.0),
    ]

    broker = BrokerSimulator(initial_cash=100_000.0, corporate_actions=actions)
    result = BacktestEngine(broker=broker).run(bars, BuyAndHold())

    print("equity curve (raw prices halve at the split, dip at the dividend):")
    print(result.performance.equity_curve().select(["timestamp", "equity", "cash"]))

    print("\ncorporate actions applied:")
    for event in broker.corporate_action_events:
        print(
            f"  {event.timestamp:%Y-%m-%d} {event.symbol} {event.action_type.value}: "
            f"qty {event.quantity_before:g} -> {event.quantity_after:g}, cash {event.cash_amount:+.2f}"
        )

    position = broker.state.positions["AAA"]
    print(f"\nfinal position: {position.quantity:g} shares @ {position.last_price:g}")
    print(f"final equity:   {result.summary()['final_equity']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
