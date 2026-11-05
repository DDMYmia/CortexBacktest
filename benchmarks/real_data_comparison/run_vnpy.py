"""vnpy (CtaStrategy) runner.

vnpy's CTA backtester is single-instrument. history_data is assigned directly
so the database layer is bypassed and only the replay is timed. Orders are
aggressive limit orders, which vnpy crosses at the next bar's open — the same
fill convention as fusion, backtrader and pyalgotrade.
"""

import contextlib
import datetime as dt
import os
import warnings
from time import perf_counter

import common

warnings.filterwarnings("ignore")

from vnpy.trader.constant import Exchange, Interval  # noqa: E402
from vnpy.trader.object import BarData  # noqa: E402
from vnpy_ctastrategy import CtaTemplate  # noqa: E402
from vnpy_ctastrategy.backtesting import BacktestingEngine  # noqa: E402

args = common.cli()
if args.symbols != 1:
    raise SystemExit("vnpy CtaStrategy is a single-instrument engine")

names, ts, cols = common.load(args.dataset, args.symbols, args.bars)
want = common.signals(cols["close"])[0]
symbol = names[0]

stamps = [dt.datetime.fromtimestamp(int(t) // 1_000_000_000, dt.UTC).replace(tzinfo=None)
          for t in ts]
o, h, low, c, v = (cols[k][0] for k in ("open", "high", "low", "close", "volume"))
history = [
    BarData(
        symbol=symbol, exchange=Exchange.NASDAQ, datetime=stamps[j],
        interval=Interval.DAILY, volume=float(v[j]), turnover=0.0, open_interest=0.0,
        open_price=float(o[j]), high_price=float(h[j]),
        low_price=float(low[j]), close_price=float(c[j]), gateway_name="BACKTEST",
    )
    for j in range(args.bars)
]

counter = {"orders": 0}
feed = common.MtfFeed(ts, cols) if args.scenario == "mtf" else None


class Sma(CtaTemplate):
    def on_init(self):
        self.cursor = 0
        self.long = False
        self.held = 0

    def on_bar(self, bar):
        i = self.cursor
        self.cursor = i + 1
        if args.scenario == "mtf":
            target = feed.target(0, i)
            delta = target - self.held
            if delta:
                self.held = target
                counter["orders"] += 1
                if delta > 0:
                    self.buy(bar.close_price * 2.0, delta)
                else:
                    self.sell(bar.close_price * 0.5, -delta)
            return
        if args.scenario == "resting":
            counter["orders"] += 1
            self.buy(round(bar.close_price * common.REST_FRACTION, 2), common.QTY)
            return
        target = bool(want[i])
        if target == self.long:
            return
        self.long = target
        counter["orders"] += 1
        if target:
            self.buy(bar.close_price * 2.0, common.QTY)
        else:
            self.sell(bar.close_price * 0.5, common.QTY)


engine = BacktestingEngine()
engine.set_parameters(
    vt_symbol=f"{symbol}.NASDAQ",
    interval=Interval.DAILY,
    start=stamps[0],
    end=stamps[-1],
    rate=0.0,
    slippage=0.0,
    size=1,
    pricetick=0.01,
    capital=int(common.INITIAL_CASH),
)
engine.add_strategy(Sma, {})
engine.history_data = history

# vnpy print()s an init banner and a progress bar; sink that so terminal I/O
# is not charged to the engine, as for qstrader.
started = perf_counter()
with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink):
    engine.run_backtesting()
elapsed = perf_counter() - started
with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink):
    engine.calculate_result()
    stats = engine.calculate_statistics(output=False)

common.emit(
    "vnpy",
    elapsed,
    counter["orders"],
    args.symbols,
    args.bars,
    "event-driven; single-instrument CTA engine; aggressive limits cross at next bar open",
    expected=common.expected_for(args.scenario, ts, cols, want),
    pnl=float(stats["total_net_pnl"]),
    convention=common.NEXT_OPEN,
    fills=len(engine.trades),
)
