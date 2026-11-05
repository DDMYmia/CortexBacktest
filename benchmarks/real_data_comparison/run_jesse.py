"""jesse runner (research backtest API, spot, zero fees).

jesse's candle engine is minute-based, so each daily bar is passed as one
1-minute candle; signals are indexed by bar, so the spacing does not matter.
Candle and route construction happen outside the timer.
"""

import contextlib
import io
import warnings
from time import perf_counter

import numpy as np

import common

warnings.filterwarnings("ignore")

import jesse.helpers as jh  # noqa: E402
from jesse.research import backtest  # noqa: E402
from jesse.strategies import Strategy  # noqa: E402

args = common.cli()
if args.scenario == "resting":
    common.unsupported("jesse", "not implemented in this harness")
names, ts, cols = common.load(args.dataset, args.symbols, args.bars)
want = common.signals(cols["close"])
feed = common.MtfFeed(ts, cols) if args.scenario == "mtf" else None
EXCHANGE = "Binance Spot"
SYMBOLS = [f"{name.replace('.', '')}-USDT" for name in names]
INDEX = {symbol: k for k, symbol in enumerate(SYMBOLS)}
counter = {"orders": 0}


def target_for(symbol: str, i: int) -> int:
    k = INDEX[symbol]
    if feed is not None:
        return feed.target(k, i)
    return common.QTY if want[k][i] else 0


class Route(Strategy):
    def before(self):
        i = self.vars.get("i", -1) + 1
        self.vars["i"] = i
        self.vars["target"] = target_for(self.symbol, i)

    def should_long(self) -> bool:
        return self.vars["target"] > 0

    def should_short(self) -> bool:
        return False

    def go_long(self):
        counter["orders"] += 1
        self.buy = self.vars["target"], self.price

    def update_position(self):
        if self.vars["target"] == 0:
            counter["orders"] += 1
            self.liquidate()

    def should_cancel_entry(self) -> bool:
        return False


start = 1_577_836_800_000  # 2020-01-01; one minute per daily bar
candles = {}
for k, symbol in enumerate(SYMBOLS):
    stamps = start + np.arange(args.bars, dtype=np.int64) * 60_000
    candles[jh.key(EXCHANGE, symbol)] = {
        "exchange": EXCHANGE, "symbol": symbol,
        "candles": np.column_stack([stamps, cols["open"][k], cols["close"][k], cols["high"][k],
                                    cols["low"][k], cols["volume"][k]]).astype(np.float64),
    }
config = {"starting_balance": common.INITIAL_CASH, "fee": 0, "type": "spot", "exchange": EXCHANGE,
          "warm_up_candles": 0}
routes = [{"exchange": EXCHANGE, "strategy": Route, "symbol": s, "timeframe": "1m"} for s in SYMBOLS]

started = perf_counter()
with contextlib.redirect_stdout(io.StringIO()):
    result = backtest(config, routes, [], candles)
elapsed = perf_counter() - started
metrics = result.get("metrics", {})
pnl = metrics.get("net_profit")

common.emit("jesse", elapsed, counter["orders"], args.symbols, args.bars,
            "event-driven research backtest; market orders at the current candle's price",
            expected=common.expected_for(args.scenario, ts, cols, want),
            pnl=pnl, convention=common.SAME_CLOSE)
