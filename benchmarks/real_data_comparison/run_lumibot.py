import logging
import os
from time import perf_counter

import pandas as pd

import common

os.environ.setdefault("IS_BACKTESTING", "true")
logging.disable(logging.CRITICAL)

from lumibot.backtesting import PandasDataBacktesting  # noqa: E402
from lumibot.entities import Asset, Data  # noqa: E402
from lumibot.strategies import Strategy  # noqa: E402

args = common.cli()
names, ts, cols = common.load(args.dataset, args.symbols, args.bars)
want = common.signals(cols["close"])
want_by_symbol = {names[s]: want[s] for s in range(args.symbols)}

index = pd.DatetimeIndex(ts.astype("datetime64[ns]")).tz_localize("America/New_York")
pandas_data = {}
assets = []
for i, sym in enumerate(names):
    asset = Asset(symbol=sym, asset_type="stock")
    frame = pd.DataFrame(
        {
            "open": cols["open"][i],
            "high": cols["high"][i],
            "low": cols["low"][i],
            "close": cols["close"][i],
            "volume": cols["volume"][i],
        },
        index=index,
    )
    pandas_data[asset] = Data(asset, frame, timestep="day")
    assets.append(asset)

counter = {"orders": 0, "fills": 0, "value": None}
feed = common.MtfFeed(ts, cols) if args.scenario == "mtf" else None


class Sma(Strategy):
    def initialize(self):
        self.sleeptime = "1D"
        self.cursor = dict.fromkeys(names, 0)
        self.long = dict.fromkeys(names, False)
        self.held = dict.fromkeys(names, 0)

    def on_filled_order(self, position, order, price, quantity, multiplier):
        counter["fills"] += 1

    def on_strategy_end(self):
        counter["value"] = self.get_portfolio_value()

    def on_trading_iteration(self):
        for asset in assets:
            sym = asset.symbol
            i = self.cursor[sym]
            self.cursor[sym] = i + 1
            if i >= args.bars:
                continue
            if args.scenario == "mtf":
                target = feed.target(names.index(sym), i)
                delta = target - self.held[sym]
                if delta:
                    self.held[sym] = target
                    counter["orders"] += 1
                    self.submit_order(self.create_order(asset, abs(delta), "buy" if delta > 0 else "sell"))
                continue
            if args.scenario == "resting":
                counter["orders"] += 1
                k = names.index(sym)
                limit = round(float(cols["close"][k][i]) * common.REST_FRACTION, 2)
                self.submit_order(self.create_order(asset, common.QTY, "buy", limit_price=limit,
                                                    time_in_force="gtc"))
                continue
            target = bool(want_by_symbol[sym][i])
            if target == self.long[sym]:
                continue
            self.long[sym] = target
            counter["orders"] += 1
            self.submit_order(
                self.create_order(asset, common.QTY, "buy" if target else "sell")
            )


started = perf_counter()
Sma.run_backtest(
    PandasDataBacktesting,
    index[0].to_pydatetime(),
    index[-1].to_pydatetime(),
    pandas_data=pandas_data,
    budget=common.INITIAL_CASH,
    benchmark_asset=None,
    analyze_backtest=False,
    show_plot=False,
    show_tearsheet=False,
    save_tearsheet=False,
    save_stats_file=False,
    show_progress_bar=False,
    quiet_logs=True,
    show_indicators=False,
)
elapsed = perf_counter() - started

common.emit(
    "lumibot",
    elapsed,
    counter["orders"],
    args.symbols,
    args.bars,
    "event-driven; analysis/tearsheet/progress disabled",
    expected=common.expected_for(args.scenario, ts, cols, want),
    pnl=None if counter["value"] is None else counter["value"] - common.INITIAL_CASH,
    convention=common.NEXT_OPEN,
    fills=counter["fills"],
)
