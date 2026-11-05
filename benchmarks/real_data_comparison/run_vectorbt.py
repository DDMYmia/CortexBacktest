from time import perf_counter

import numpy as np
import common

import vectorbt as vbt
import pandas as pd

args = common.cli()
names, ts, cols = common.load(args.dataset, args.symbols, args.bars)
want = common.signals(cols["close"])

if args.scenario == "resting":
    common.unsupported("vectorbt", "no resting order book: orders exist only on the bar they are placed")

index = pd.DatetimeIndex(ts.astype("datetime64[ns]"))
close = pd.DataFrame(cols["close"].T, index=index, columns=names)
opens = pd.DataFrame(cols["open"].T, index=index, columns=names)
# The decision on bar t is traded at the open of bar t+1: shift the signal one
# bar and trade the change in holdings. Positions are marked at the close.
def sizes(targets):
    held = np.vstack([np.zeros((1, targets.shape[0])), targets.T[:-1].astype(float)])
    previous = np.vstack([np.zeros((1, held.shape[1])), held[:-1]])
    return pd.DataFrame(held - previous, index=index, columns=names)


size = sizes(want * common.QTY)

# Warm numba so JIT compilation is not charged to the measured run.
vbt.Portfolio.from_orders(close.iloc[:60, :2], size.iloc[:60, :2], price=opens.iloc[:60, :2],
                          init_cash=common.INITIAL_CASH, fees=0.0, freq="1D", cash_sharing=True,
                          group_by=True).total_profit()

started = perf_counter()
if args.scenario == "mtf":  # vectorized engines get no callback: the strategy runs as a timed pre-pass
    size = sizes(common.mtf_targets(ts, cols))
pf = vbt.Portfolio.from_orders(close, size, price=opens, init_cash=common.INITIAL_CASH, fees=0.0,
                               freq="1D", cash_sharing=True, group_by=True)
n = len(pf.orders.records)
pnl = float(pf.total_profit())
elapsed = perf_counter() - started

common.emit("vectorbt", elapsed, n, args.symbols, args.bars,
            "vectorized; whole-column compute, no per-bar broker loop; numba pre-warmed",
            expected=common.expected_for(args.scenario, ts, cols, want), pnl=pnl, convention=common.NEXT_OPEN, fills=n)
