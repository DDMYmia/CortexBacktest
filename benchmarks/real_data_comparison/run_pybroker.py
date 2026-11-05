from time import perf_counter

import pandas as pd

import common

import pybroker
from pybroker import Strategy, StrategyConfig
from pybroker.common import PriceType

pybroker.disable_logging()

args = common.cli()
if args.scenario == "resting":
    common.unsupported("pybroker", "limit orders are not kept open across bars (no GTC resting order)")
names, ts, cols = common.load(args.dataset, args.symbols, args.bars)
want = common.signals(cols["close"])
want_by_symbol = {names[s]: want[s] for s in range(args.symbols)}

index = pd.DatetimeIndex(ts.astype("datetime64[ns]"))
df = pd.concat(
    [
        pd.DataFrame(
            {
                "symbol": sym,
                "date": index,
                "open": cols["open"][i],
                "high": cols["high"][i],
                "low": cols["low"][i],
                "close": cols["close"][i],
                "volume": cols["volume"][i],
            }
        )
        for i, sym in enumerate(names)
    ],
    ignore_index=True,
)

cursor = dict.fromkeys(names, 0)
state = dict.fromkeys(names, False)
counter = {"orders": 0}


index_of = {name: k for k, name in enumerate(names)}
held = dict.fromkeys(names, 0)
feed = common.MtfFeed(ts, cols) if args.scenario == "mtf" else None


def execute_mtf(ctx):
    sym = ctx.symbol
    i = cursor[sym]
    cursor[sym] = i + 1
    target = feed.target(index_of[sym], i)
    delta = target - held[sym]
    if not delta:
        return
    held[sym] = target
    counter["orders"] += 1
    if delta > 0:
        ctx.buy_shares = delta
        ctx.buy_fill_price = PriceType.OPEN
    else:
        ctx.sell_shares = -delta
        ctx.sell_fill_price = PriceType.OPEN


def execute(ctx):
    sym = ctx.symbol
    i = cursor[sym]
    cursor[sym] = i + 1
    target = bool(want_by_symbol[sym][i])
    if target == state[sym]:
        return
    state[sym] = target
    counter["orders"] += 1
    if target:
        ctx.buy_shares = common.QTY
        ctx.buy_fill_price = PriceType.OPEN
    else:
        ctx.sell_shares = common.QTY
        ctx.sell_fill_price = PriceType.OPEN


config = StrategyConfig(
    initial_cash=common.INITIAL_CASH,
    buy_delay=1,
    sell_delay=1,
    exit_on_last_bar=False,
    round_fill_price=False,  # fill at the bar's price, not rounded to cents
    round_test_result=False,  # report P&L unrounded too
)
strategy = Strategy(df, index[0], index[-1], config)
strategy.set_before_exec(None) if hasattr(strategy, "set_before_exec") else None
strategy.add_execution(execute_mtf if args.scenario == "mtf" else execute, names)

started = perf_counter()
result = strategy.backtest(calc_bootstrap=False)
elapsed = perf_counter() - started
pnl = float(result.portfolio["market_value"].iloc[-1]) - common.INITIAL_CASH

common.emit(
    "pybroker",
    elapsed,
    counter["orders"],
    args.symbols,
    args.bars,
    "event-driven; buy_delay=1 at the next open, analysis/bootstrap disabled",
    expected=common.expected_for(args.scenario, ts, cols, want),
    pnl=pnl,
    convention=common.NEXT_OPEN,
    fills=int(len(result.orders)),
)
