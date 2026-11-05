import sys
from pathlib import Path
from time import perf_counter

import common

# Derived, not hardcoded: an absolute path here silently imports the main
# checkout even when this harness is run from another worktree.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from cortexbacktest import (  # noqa: E402
    BacktestEngine,
    Bar,
    BrokerSimulator,
    FillPrice,
    Order,
    OrderType,
    Side,
    Strategy,
)

args = common.cli()
names, ts, cols = common.load(args.dataset, args.symbols, args.bars)
want = common.signals(cols["close"])

import pandas as pd  # noqa: E402

stamps = [t.to_pydatetime() for t in pd.DatetimeIndex(ts.astype("datetime64[ns]"), tz="UTC")]

o, h, low, c, v = (cols[k] for k in ("open", "high", "low", "close", "volume"))
bars = [
    Bar(names[s], stamps[b], o[s, b], h[s, b], low[s, b], c[s, b], v[s, b])
    for b in range(args.bars)
    for s in range(args.symbols)
]
want_by_symbol = {names[s]: want[s] for s in range(args.symbols)}


class Sma(Strategy):
    def __init__(self):
        self.cursor = dict.fromkeys(want_by_symbol, 0)
        self.long = dict.fromkeys(want_by_symbol, False)
        self.orders = 0

    def on_bar(self, bar):
        i = self.cursor[bar.symbol]
        self.cursor[bar.symbol] = i + 1
        target = bool(want_by_symbol[bar.symbol][i])
        if target == self.long[bar.symbol]:
            return []
        self.long[bar.symbol] = target
        self.orders += 1
        return [Order(bar.symbol, bar.timestamp, Side.BUY if target else Side.SELL, common.QTY)]


class MtfStrategy(Strategy):
    """Computes the multi-timeframe target inside the callback, then trades the change."""

    def __init__(self):
        self.index = {name: k for k, name in enumerate(names)}
        self.cursor = dict.fromkeys(names, 0)
        self.held = dict.fromkeys(names, 0)
        self.orders = 0

    def on_bar(self, bar):
        i = self.cursor[bar.symbol]
        self.cursor[bar.symbol] = i + 1
        target = feed.target(self.index[bar.symbol], i)
        delta = target - self.held[bar.symbol]
        if not delta:
            return []
        self.held[bar.symbol] = target
        self.orders += 1
        return [Order(bar.symbol, bar.timestamp, Side.BUY if delta > 0 else Side.SELL, abs(delta))]


class Resting(Strategy):
    orders = 0

    def on_bar(self, bar):
        Resting.orders += 1
        return [Order(bar.symbol, bar.timestamp, Side.BUY, common.QTY, order_type=OrderType.LIMIT,
                      limit_price=round(bar.close * common.REST_FRACTION, 2))]


TIMING = {  # convention -> (trade_on_close, fill price)
    common.NEXT_OPEN: (False, FillPrice.OPEN),
    common.NEXT_CLOSE: (False, FillPrice.CLOSE),
    common.SAME_CLOSE: (True, FillPrice.CLOSE),
}
on_close, fill_price = TIMING[args.convention]
feed = common.MtfFeed(ts, cols) if args.scenario == "mtf" else None
strategy = {"sma": Sma, "resting": Resting, "mtf": MtfStrategy}[args.scenario]()
engine = BacktestEngine(
    broker=BrokerSimulator(initial_cash=common.INITIAL_CASH, fill_price=fill_price, allow_short=False),
    trade_on_close=on_close,
)
started = perf_counter()
result = engine.run(bars, strategy)
elapsed = perf_counter() - started

common.emit("fusion", elapsed, strategy.orders, args.symbols, args.bars,
            f"event-driven; {args.convention} fills",
            expected=common.expected_for(args.scenario, ts, cols, want),
            pnl=result.summary()["final_equity"] - common.INITIAL_CASH,
            convention=args.convention, fills=result.fill_count)
