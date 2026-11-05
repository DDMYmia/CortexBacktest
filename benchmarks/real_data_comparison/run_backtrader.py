from time import perf_counter

import common

import backtrader as bt

args = common.cli()
names, ts, cols = common.load(args.dataset, args.symbols, args.bars)
want = common.signals(cols["close"])
data = common.frames(args.dataset, args.symbols, args.bars)


class Sma(bt.Strategy):
    def __init__(self):
        self.cursor = [0] * len(self.datas)
        self.long = [False] * len(self.datas)
        self.orders = 0

    def next(self):
        for i, d in enumerate(self.datas):
            j = self.cursor[i]
            self.cursor[i] = j + 1
            target = bool(want[i][j])
            if target == self.long[i]:
                continue
            self.long[i] = target
            self.orders += 1
            if target:
                self.buy(data=d, size=common.QTY)
            else:
                self.sell(data=d, size=common.QTY)


class MtfStrategy(bt.Strategy):
    def __init__(self):
        self.cursor = [0] * len(self.datas)
        self.held = [0] * len(self.datas)
        self.orders = 0

    def next(self):
        for i, d in enumerate(self.datas):
            j = self.cursor[i]
            self.cursor[i] = j + 1
            target = feed.target(i, j)
            delta = target - self.held[i]
            if not delta:
                continue
            self.held[i] = target
            self.orders += 1
            if delta > 0:
                self.buy(data=d, size=delta)
            else:
                self.sell(data=d, size=-delta)


class Resting(bt.Strategy):
    def __init__(self):
        self.orders = 0
        self.fills = 0

    def notify_order(self, order):
        if order.status == order.Completed:
            self.fills += 1

    def next(self):
        for d in self.datas:
            self.orders += 1
            self.buy(data=d, size=common.QTY, exectype=bt.Order.Limit,
                     price=round(d.close[0] * common.REST_FRACTION, 2))


class CountFills(bt.Analyzer):
    def start(self):
        self.fills = 0

    def notify_order(self, order):
        if order.status == order.Completed:
            self.fills += 1


cerebro = bt.Cerebro(stdstats=False, runonce=False)
cerebro.broker.setcash(common.INITIAL_CASH)
cerebro.broker.set_coc(False)  # fill at next bar open
for sym in names:
    cerebro.adddata(bt.feeds.PandasData(dataname=data[sym]), name=sym)
feed = common.MtfFeed(ts, cols) if args.scenario == "mtf" else None
cerebro.addstrategy({"sma": Sma, "resting": Resting, "mtf": MtfStrategy}[args.scenario])
cerebro.addanalyzer(CountFills, _name="fills")

started = perf_counter()
strategies = cerebro.run()
elapsed = perf_counter() - started

common.emit("backtrader", elapsed, strategies[0].orders, args.symbols, args.bars,
            "event-driven; next-bar-open fills",
            expected=common.expected_for(args.scenario, ts, cols, want),
            pnl=cerebro.broker.getvalue() - common.INITIAL_CASH, convention=common.NEXT_OPEN,
            fills=strategies[0].analyzers.fills.fills)
