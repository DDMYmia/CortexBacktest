from time import perf_counter

import common

from backtesting import Backtest, Strategy
import backtesting.backtesting as _bt_mod

# Disable the tqdm progress bar so terminal reporting is not charged to the
# engine. Same courtesy extended to nautilus (logging) and vectorbt (JIT warm-up).
_bt_mod._tqdm = lambda seq, **_: seq

args = common.cli()
if args.symbols != 1:
    raise SystemExit("backtesting.py is a single-asset engine")

names, ts, cols = common.load(args.dataset, args.symbols, args.bars)
want = common.signals(cols["close"])[0]
frame = common.frames(args.dataset, args.symbols, args.bars)[names[0]]


FILLS = [0]


class Sma(Strategy):
    def init(self):
        self.long = False
        self.n_orders = 0

    def next(self):
        # Every open trade had an entry fill; every closed one an entry and an exit.
        FILLS[0] = len(self.trades) + 2 * len(self.closed_trades)
        # backtesting.py first calls next() on the second bar, so index by the bar
        # being processed, not by a call counter.
        i = len(self.data) - 1
        target = bool(want[i])
        if target == self.long:
            return
        self.long = target
        self.n_orders += 1
        if target:
            self.buy(size=common.QTY)
        else:
            self.position.close()


class MtfStrategy(Strategy):
    def init(self):
        self.held = 0
        self.n_orders = 0

    def next(self):
        FILLS[0] = len(self.trades) + 2 * len(self.closed_trades)
        target = feed.target(0, len(self.data) - 1)
        delta = target - self.held
        if not delta:
            return
        self.held = target
        self.n_orders += 1
        if delta > 0:
            self.buy(size=delta)
        else:  # targets only go flat -> size -> flat
            self.position.close()


class Resting(Strategy):
    def init(self):
        self.n_orders = 0

    def next(self):
        FILLS[0] = len(self.trades) + 2 * len(self.closed_trades)
        self.n_orders += 1
        self.buy(size=common.QTY, limit=round(float(self.data.Close[-1]) * common.REST_FRACTION, 2))


feed = common.MtfFeed(ts, cols) if args.scenario == "mtf" else None
backtest = Backtest(frame, {"sma": Sma, "resting": Resting, "mtf": MtfStrategy}[args.scenario], cash=common.INITIAL_CASH,
                    commission=0.0, trade_on_close=False, exclusive_orders=False, finalize_trades=False)
started = perf_counter()
stats = backtest.run()
elapsed = perf_counter() - started

common.emit("backtesting.py", elapsed, stats["_strategy"].n_orders, args.symbols, args.bars,
            "event-driven; single-asset engine only",
            expected=common.expected_for(args.scenario, ts, cols, want),
            pnl=float(stats["Equity Final [$]"]) - common.INITIAL_CASH, convention=common.NEXT_OPEN,
            fills=FILLS[0])
