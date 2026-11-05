"""freqtrade runner: in-process Backtesting on local daily candles.

freqtrade is a crypto engine and validates pairs against an exchange's market
list, so each equity is relabelled as a real OKX USDT spot pair (fine price and
amount precision) and its bars are written as that pair's local 1d data. The
market list is fetched once, outside the timer. Signals on a candle fill at the
next candle's open; the stake is target shares x that open, so the position is
the target share count. Open trades are closed at the end at the last close.
Timed: the strategy's indicator pass plus the backtest loop.
"""

import json
import logging
import shutil
import sys
import warnings
from pathlib import Path
from time import perf_counter

import pandas as pd

import common

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

args = common.cli()
if args.scenario == "resting":
    common.unsupported("freqtrade", "entries are signal-driven; no GTC resting limit book per bar")
names, ts, cols = common.load(args.dataset, args.symbols, args.bars)

import ccxt  # noqa: E402

markets = ccxt.okx().load_markets()
pairs = sorted(s for s, m in markets.items()
               if m.get("spot") and m.get("active") and m.get("quote") == "USDT"
               and (m["precision"].get("amount") or 1) <= 1e-4 and (m["precision"].get("price") or 1) <= 1e-4
               # Orders below the exchange minimum are rejected; the targets go down to 1 share.
               and (m["limits"]["amount"].get("min") or 0) <= 1)
if len(pairs) < len(names):
    common.unsupported("freqtrade", f"only {len(pairs)} OKX USDT pairs with fine precision for {len(names)} symbols")
pairs = pairs[: len(names)]

HERE = Path(__file__).resolve().parent
work = HERE / ".freqtrade" / f"{args.dataset.split('.')[0]}_{args.symbols}x{args.bars}_{args.scenario}"
shutil.rmtree(work, ignore_errors=True)
data_dir = work / "data" / "okx"
data_dir.mkdir(parents=True)
# Consecutive calendar days: freqtrade fills weekend/holiday gaps with synthetic candles,
# which would shift every bar. Signals are indexed by bar, so the labels do not matter.
dates = pd.date_range("2000-01-01", periods=args.bars, freq="D", tz="UTC")
for k, pair in enumerate(pairs):
    pd.DataFrame({"date": dates, "open": cols["open"][k], "high": cols["high"][k], "low": cols["low"][k],
                  "close": cols["close"][k], "volume": cols["volume"][k]}
                 ).to_feather(data_dir / f"{pair.replace('/', '_')}-1d.feather")

strategy_dir = work / "strategies"
strategy_dir.mkdir()
(strategy_dir / "harness_strategy.py").write_text(f'''
import sys
sys.path.insert(0, {str(HERE)!r})
import numpy as np
from freqtrade.strategy import IStrategy
import common

ARGS = {{"dataset": {args.dataset!r}, "symbols": {args.symbols}, "bars": {args.bars}, "scenario": {args.scenario!r}}}
PAIRS = {pairs!r}
NAMES, TS, COLS = common.load(ARGS["dataset"], ARGS["symbols"], ARGS["bars"])
WANT = common.signals(COLS["close"])
FEED = common.MtfFeed(TS, COLS) if ARGS["scenario"] == "mtf" else None
TARGETS = {{}}


class Harness(IStrategy):
    timeframe = "1d"
    minimal_roi = {{"0": 1000.0}}
    stoploss = -0.99
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = True
    startup_candle_count = 0
    process_only_new_candles = False

    def populate_indicators(self, dataframe, metadata):
        k = PAIRS.index(metadata["pair"])
        if FEED is not None:
            target = [FEED.target(k, i) for i in range(len(dataframe))]
        else:
            target = list(np.where(WANT[k][: len(dataframe)], common.QTY, 0))
        dataframe["target"] = target
        TARGETS[metadata["pair"]] = dict(zip(dataframe["date"], target))
        return dataframe

    def populate_entry_trend(self, dataframe, metadata):
        previous = dataframe["target"].shift(1).fillna(0)
        dataframe.loc[(dataframe["target"] > 0) & (previous == 0), "enter_long"] = 1
        return dataframe

    def populate_exit_trend(self, dataframe, metadata):
        previous = dataframe["target"].shift(1).fillna(0)
        dataframe.loc[(dataframe["target"] == 0) & (previous > 0), "exit_long"] = 1
        return dataframe

    def custom_stake_amount(self, pair, current_time, current_rate, proposed_stake, min_stake, max_stake,
                            leverage, entry_tag, side, **kwargs):
        signal = self.dp.get_analyzed_dataframe(pair, self.timeframe)[0]
        rows = signal[signal["date"] < current_time]
        return float(rows["target"].iloc[-1]) * current_rate
''')
config_path = work / "config.json"
config_path.write_text(json.dumps({
    "max_open_trades": len(pairs), "stake_currency": "USDT", "stake_amount": "unlimited", "tradable_balance_ratio": 1.0,
    "dry_run": True, "dry_run_wallet": common.INITIAL_CASH, "timeframe": "1d", "fee": 0.0,
    "exchange": {"name": "okx", "key": "", "secret": "", "pair_whitelist": pairs},
    "pairlists": [{"method": "StaticPairList"}],
    "entry_pricing": {"price_side": "same", "use_order_book": False},
    "exit_pricing": {"price_side": "same", "use_order_book": False},
    "user_data_dir": str(work), "datadir": str(data_dir), "dataformat_ohlcv": "feather",
}))

from freqtrade.configuration import Configuration  # noqa: E402
from freqtrade.enums import RunMode  # noqa: E402
from freqtrade.optimize.backtesting import Backtesting  # noqa: E402

config = Configuration({"config": [str(config_path)], "strategy": "Harness", "strategy_path": str(strategy_dir),
                        "export": "none", "fee": 0.0}, RunMode.BACKTEST).get_config()
backtesting = Backtesting(config)
data, _ = backtesting.load_bt_data()
backtesting._set_strategy(backtesting.strategylist[0])  # as Backtesting.backtest_one_strategy does

started = perf_counter()
processed = backtesting.strategy.advise_all_indicators(data)
min_date = min(frame["date"].min() for frame in processed.values())
max_date = max(frame["date"].max() for frame in processed.values())
result = backtesting.backtest(processed=processed, start_date=min_date, end_date=max_date)
elapsed = perf_counter() - started

trades = result["results"]
want = common.signals(cols["close"])
# freqtrade closes trades still open at the end at the last candle's open; every other
# engine (and the ledger) marks them at the last close, so re-mark those exits there.
last_close = {pair: float(cols["close"][k][-1]) for k, pair in enumerate(pairs)}
forced = trades["exit_reason"] == "force_exit"
pnl = float(trades["profit_abs"].sum())
if forced.any():
    marks = trades.loc[forced, "pair"].astype(object).map(last_close).astype(float)
    pnl += float((trades.loc[forced, "amount"] * (marks - trades.loc[forced, "close_rate"])).sum())
orders = int(len(trades) * 2 - (trades["exit_reason"] == "force_exit").sum()) if len(trades) else 0
common.emit("freqtrade", elapsed, orders, args.symbols, args.bars,
            "relabelled as OKX USDT pairs; next-candle-open fills; trades open at the end re-marked at the last close",
            expected=common.expected_for(args.scenario, ts, cols, want),
            pnl=pnl, convention=common.NEXT_OPEN)
sys.stdout.flush()
