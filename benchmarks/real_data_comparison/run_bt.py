from time import perf_counter

import numpy as np
import pandas as pd

import common

import bt

args = common.cli()
if args.scenario != "sma":
    common.unsupported("bt", "weight-based rebalancer: no per-symbol orders or share-count targets")
names, ts, cols = common.load(args.dataset, args.symbols, args.bars)
want = common.signals(cols["close"])

index = pd.DatetimeIndex(ts.astype("datetime64[ns]"))
prices = pd.DataFrame(cols["close"].T, index=index, columns=names)

# bt is a weight-based portfolio engine with no per-symbol share orders, so the
# same signal is expressed as an equal weight across whatever is long, shifted
# one bar so the weight applies after the signal. Prices are closes: bt has no
# next-bar-open fill convention. Both differences are noted in the output.
w = pd.DataFrame(want.T, index=index, columns=names).shift(1).fillna(False).astype(bool)
counts = w.sum(axis=1).replace(0, np.nan)
target = w.div(counts, axis=0).fillna(0.0) * 0.99  # cash buffer; bt refuses a fully invested rebalance
changes = int(w.astype(int).diff().abs().sum().sum())

strategy = bt.Strategy("sma", [bt.algos.WeighTarget(target), bt.algos.Rebalance()])
backtest = bt.Backtest(
    strategy,
    prices,
    initial_capital=1_000_000.0,  # bt aborts at 1e9: its share-rounding loop fails to converge
    integer_positions=False,
)

started = perf_counter()
result = bt.run(backtest)
elapsed = perf_counter() - started
pnl = float(result.prices.iloc[-1, 0] / 100.0 * 1_000_000.0 - 1_000_000.0)

common.emit(
    "bt",
    elapsed,
    changes,
    args.symbols,
    args.bars,
    "weight-based portfolio engine; equal-weight proxy, close fills (no next-open convention), 1e6 capital",
    expected=common.expected_orders(want),
    pnl=pnl,
    convention="weight-based",
)
