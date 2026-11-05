"""QSTrader runner.

QSTrader is a weight-based portfolio engine driven by an alpha model, and it
reads daily bars from CSV files on disk. The CSVs are written and the data
handler is constructed outside the timed region; only session.run() is timed.
Like bt, it has no per-symbol share order, so the same signal is expressed as
an equal weight over whatever is long.
"""

import contextlib
import logging
import os
import shutil
import warnings
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import pytz

import common

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)  # qstrader logs every order to stdout

from qstrader.alpha_model.alpha_model import AlphaModel  # noqa: E402
from qstrader.asset.equity import Equity  # noqa: E402
from qstrader.asset.universe.static import StaticUniverse  # noqa: E402
from qstrader.data.backtest_data_handler import BacktestDataHandler  # noqa: E402
from qstrader.data.daily_bar_csv import CSVDailyBarDataSource  # noqa: E402
from qstrader.trading.backtest import BacktestTradingSession  # noqa: E402

args = common.cli()
if args.scenario != "sma":
    common.unsupported("qstrader", "weight-based rebalancer: no per-symbol orders or share-count targets")
names, ts, cols = common.load(args.dataset, args.symbols, args.bars)
want = common.signals(cols["close"])
index = pd.DatetimeIndex(ts.astype("datetime64[ns]"))

csv_dir = Path(__file__).resolve().parent / f".qs_{args.symbols}x{args.bars}"
shutil.rmtree(csv_dir, ignore_errors=True)
csv_dir.mkdir(parents=True)
for i, sym in enumerate(names):
    pd.DataFrame(
        {
            "Date": index.strftime("%Y-%m-%d"),
            "Open": cols["open"][i],
            "High": cols["high"][i],
            "Low": cols["low"][i],
            "Close": cols["close"][i],
            "Adj Close": cols["close"][i],
            "Volume": cols["volume"][i].astype("int64"),
        }
    ).to_csv(csv_dir / f"{sym}.csv", index=False)

asset_names = [f"EQ:{s}" for s in names]
universe = StaticUniverse(asset_names)
source = CSVDailyBarDataSource(str(csv_dir), Equity, csv_symbols=names, adjust_prices=False)
handler = BacktestDataHandler(universe, data_sources=[source])

row_of = {d.date(): i for i, d in enumerate(index)}
counter = {"orders": 0}
prev = np.zeros(args.symbols, bool)


class Sma(AlphaModel):
    def __call__(self, dt):
        i = row_of.get(dt.date())
        if i is None:
            return dict.fromkeys(asset_names, 0.0)
        global prev
        w = want[:, i]
        counter["orders"] += int((w != prev).sum())
        prev = w
        n = int(w.sum())
        if n == 0:
            return dict.fromkeys(asset_names, 0.0)
        return {a: (1.0 / n if w[k] else 0.0) for k, a in enumerate(asset_names)}


session = BacktestTradingSession(
    index[0].tz_localize(pytz.UTC),
    index[-1].tz_localize(pytz.UTC),
    universe,
    Sma(),
    initial_cash=common.INITIAL_CASH,
    rebalance="daily",
    long_only=True,
    cash_buffer_percentage=0.01,
    data_handler=handler,
)

# qstrader print()s every rebalance and every order; sink that to devnull so
# terminal I/O is not charged to the engine (same courtesy as nautilus logging).
started = perf_counter()
with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink):
    session.run(results=False)
elapsed = perf_counter() - started

common.emit(
    "qstrader",
    elapsed,
    counter["orders"],
    args.symbols,
    args.bars,
    "weight-based portfolio engine; equal-weight proxy; CSVs written outside the timer",
    expected=common.expected_orders(want),
    pnl=float(session.get_equity_curve()["Equity"].iloc[-1]) - common.INITIAL_CASH,
    convention="weight-based",
)
