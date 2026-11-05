"""cipher-bt runner.

Single-asset, session-based engine. Its public Cipher.run() loads data from a
source inside the run, so this drives the Trader directly with a DataFrame
built outside the timer — otherwise CSV parsing would be charged to the engine
and no other entry pays that. Fills are at close: cipher has no next-bar-open
convention.
"""

from time import perf_counter

import pandas as pd

import common

from cipher.models import Datas
from cipher.strategy import Strategy
from cipher.trader import Trader

args = common.cli()
if args.scenario != "sma":
    common.unsupported("cipher-bt", "not implemented in this harness")
if args.symbols != 1:
    raise SystemExit("cipher-bt is a single-asset engine")

names, ts, cols = common.load(args.dataset, args.symbols, args.bars)
want = common.signals(cols["close"])[0]

index = pd.DatetimeIndex(ts.astype("datetime64[ns]"))
prev = pd.Series(want, index=index).shift(1).fillna(False).astype(bool)
now = pd.Series(want, index=index)
frame = pd.DataFrame(
    {
        "open": cols["open"][0],
        "high": cols["high"][0],
        "low": cols["low"][0],
        "close": cols["close"][0],
        "volume": cols["volume"][0],
        "entry": (now & ~prev).to_numpy(),
        "exit": (~now & prev).to_numpy(),
    },
    index=index,
)
expected = int(frame["entry"].sum() + frame["exit"].sum())


class Sma(Strategy):
    def compose(self):
        return frame

    def on_entry(self, row, session):
        session.position += common.QTY

    def on_exit(self, row, session):
        session.position = 0


started = perf_counter()
Trader(Datas([frame]), Sma()).run()
elapsed = perf_counter() - started

common.emit(
    "cipher-bt",
    elapsed,
    expected,
    args.symbols,
    args.bars,
    "event-driven; single-asset; pandas iterrows loop; close fills, not next-open",
    expected=common.expected_orders(want.reshape(1, -1)),
)
