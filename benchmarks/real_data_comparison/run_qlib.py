"""qlib runner: qlib.backtest with a custom order-level strategy.

qlib reads its own binary layout, written here outside the timer: a day
calendar, an instrument list, and one float32 file per field and symbol (a
start index into the calendar, then values). float32 storage rounds prices to
~7 significant digits, which shows up as cents in P&L.

The strategy is called at the start of each trading step and its orders fill at
that step's open (`deal_price="open"`), so it decides from the previous bar:
the next-bar-open convention.
"""

import contextlib
import io
import logging
import shutil
import warnings
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

import common

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

import qlib  # noqa: E402
from qlib.backtest import backtest as qlib_backtest  # noqa: E402
from qlib.backtest.decision import Order, OrderDir, TradeDecisionWO  # noqa: E402
from qlib.backtest.executor import SimulatorExecutor  # noqa: E402
from qlib.strategy.base import BaseStrategy  # noqa: E402

args = common.cli()
if args.scenario == "resting":
    common.unsupported("qlib", "orders live for one trading step; no GTC resting book")
names, ts, cols = common.load(args.dataset, args.symbols, args.bars)
want = common.signals(cols["close"])
feed = common.MtfFeed(ts, cols) if args.scenario == "mtf" else None
days = pd.DatetimeIndex(ts.astype("datetime64[ns]")).normalize()
CODES = [f"S{k:04d}" for k in range(len(names))]  # qlib lower-cases ids; avoid clashes

root = Path(__file__).resolve().parent / f".qlib_{args.dataset.split('.')[0]}_{args.symbols}x{args.bars}"
shutil.rmtree(root, ignore_errors=True)
(root / "calendars").mkdir(parents=True)
(root / "instruments").mkdir()
# One trailing session: qlib ends each step at the next calendar day.
calendar = list(days) + [days[-1] + pd.offsets.BDay(1)]
(root / "calendars" / "day.txt").write_text("\n".join(d.strftime("%Y-%m-%d") for d in calendar) + "\n")
(root / "instruments" / "all.txt").write_text(
    "".join(f"{code}\t{days[0]:%Y-%m-%d}\t{days[-1]:%Y-%m-%d}\n" for code in CODES))
for k, code in enumerate(CODES):
    folder = root / "features" / code.lower()
    folder.mkdir(parents=True)
    close = cols["close"][k]
    fields = {"open": cols["open"][k], "high": cols["high"][k], "low": cols["low"][k], "close": close,
              "volume": cols["volume"][k], "factor": np.ones_like(close),
              "change": np.concatenate([[np.nan], close[1:] / close[:-1] - 1.0])}
    for field, values in fields.items():
        np.hstack([[0.0], values]).astype("<f4").tofile(folder / f"{field}.day.bin")

with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    qlib.init(provider_uri=str(root), region="us", kernels=1)

counter = {"orders": 0}


class Rules(BaseStrategy):
    """Per step: compute every symbol's target from the previous bar, trade the change."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.held = [0] * len(CODES)

    def generate_trade_decision(self, execute_result=None):
        step = self.trade_calendar.get_trade_step()
        start, end = self.trade_calendar.get_step_time(step)
        orders = []
        if step >= 1:
            i = step - 1
            for k, code in enumerate(CODES):
                target = feed.target(k, i) if feed is not None else (common.QTY if want[k][i] else 0)
                delta = target - self.held[k]
                if delta:
                    self.held[k] = target
                    counter["orders"] += 1
                    orders.append(Order(stock_id=code, amount=float(abs(delta)), start_time=start, end_time=end,
                                        direction=OrderDir.BUY if delta > 0 else OrderDir.SELL))
        return TradeDecisionWO(orders, self)


executor = SimulatorExecutor(time_per_step="day", generate_portfolio_metrics=True, verbose=False)
exchange = {"freq": "day", "deal_price": "open", "open_cost": 0.0, "close_cost": 0.0, "min_cost": 0.0,
            "limit_threshold": None, "trade_unit": None, "codes": CODES}
started = perf_counter()
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    portfolio, indicators = qlib_backtest(start_time=days[0], end_time=days[-1], strategy=Rules(),
                                          executor=executor, benchmark=CODES[0], account=common.INITIAL_CASH,
                                          exchange_kwargs=exchange)
elapsed = perf_counter() - started
report = portfolio["1day"][0]
pnl = float(report["account"].iloc[-1]) - common.INITIAL_CASH

common.emit("qlib", elapsed, counter["orders"], args.symbols, args.bars,
            "qlib.backtest, order-level strategy; fills at the step's open; float32 price storage",
            expected=common.expected_for(args.scenario, ts, cols, want),
            pnl=pnl, convention=common.NEXT_OPEN)
