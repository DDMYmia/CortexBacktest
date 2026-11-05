import datetime as dt
from time import perf_counter

import common

from pyalgotrade import bar, strategy
from pyalgotrade.barfeed import membf

args = common.cli()
names, ts, cols = common.load(args.dataset, args.symbols, args.bars)
want = common.signals(cols["close"])
want_by_symbol = {names[s]: want[s] for s in range(args.symbols)}

stamps = [dt.datetime.utcfromtimestamp(int(t) // 1_000_000_000) for t in ts]

class MemFeed(membf.BarFeed):
    def barsHaveAdjClose(self):
        return False


feed = MemFeed(bar.Frequency.DAY)
for i, sym in enumerate(names):
    o, h, low, c, v = (cols[k][i] for k in ("open", "high", "low", "close", "volume"))
    feed.addBarsFromSequence(
        sym,
        [
            bar.BasicBar(stamps[j], o[j], h[j], low[j], c[j], v[j], c[j], bar.Frequency.DAY)
            for j in range(args.bars)
        ],
    )


feed_mtf = common.MtfFeed(ts, cols) if args.scenario == "mtf" else None


class Sma(strategy.BacktestingStrategy):
    def __init__(self, feed):
        super().__init__(feed, common.INITIAL_CASH)
        self.cursor = dict.fromkeys(names, 0)
        self.long = dict.fromkeys(names, False)
        self.n_orders = 0
        self.n_fills = 0
        self.held = dict.fromkeys(names, 0)
        self.getBroker().setAllowNegativeCash(True)
        # Default fills cap at 25% of bar volume; the other engines and the ledger have no cap.
        self.getBroker().getFillStrategy().setVolumeLimit(None)

    def onOrderUpdated(self, order):
        if order.isFilled():
            self.n_fills += 1

    def onBars(self, bars):
        for sym in bars.getInstruments():
            i = self.cursor[sym]
            self.cursor[sym] = i + 1
            if args.scenario == "mtf":
                target = feed_mtf.target(names.index(sym), i)
                delta = target - self.held[sym]
                if delta:
                    self.held[sym] = target
                    self.n_orders += 1
                    self.marketOrder(sym, delta, onClose=False)
                continue
            if args.scenario == "resting":
                self.n_orders += 1
                price = round(bars[sym].getClose() * common.REST_FRACTION, 2)
                self.limitOrder(sym, price, common.QTY, goodTillCanceled=True)
                continue
            target = bool(want_by_symbol[sym][i])
            if target == self.long[sym]:
                continue
            self.long[sym] = target
            self.n_orders += 1
            if target:
                self.marketOrder(sym, common.QTY, onClose=False)
            else:
                self.marketOrder(sym, -common.QTY, onClose=False)


algo = Sma(feed)
started = perf_counter()
algo.run()
elapsed = perf_counter() - started

common.emit(
    "pyalgotrade",
    elapsed,
    algo.n_orders,
    args.symbols,
    args.bars,
    "event-driven; market orders fill at next bar open",
    expected=common.expected_for(args.scenario, ts, cols, want),
    pnl=algo.getBroker().getEquity() - common.INITIAL_CASH,
    convention=common.NEXT_OPEN,
    fills=algo.n_fills,
)
