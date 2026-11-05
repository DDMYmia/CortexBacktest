"""Zipline runner.

Zipline cannot take bars from memory: it requires an ingested bundle on disk.
Ingestion is data preparation and happens outside the timed region, exactly as
DataFrame/feed construction does for every other engine. Only run_algorithm is
timed.
"""

import os
import shutil
import warnings
from pathlib import Path
from time import perf_counter

import pandas as pd

import common

warnings.filterwarnings("ignore")
ZIPLINE_ROOT = Path(__file__).resolve().parent / ".zipline"
os.environ["ZIPLINE_ROOT"] = str(ZIPLINE_ROOT)

from zipline import run_algorithm  # noqa: E402
from zipline.api import commission, order, set_commission, set_slippage, slippage  # noqa: E402
from zipline.data import bundles  # noqa: E402
from zipline.finance.execution import LimitOrder  # noqa: E402

args = common.cli()
names, ts, cols = common.load(args.dataset, args.symbols, args.bars)
want = common.signals(cols["close"])
index = pd.DatetimeIndex(ts.astype("datetime64[ns]"))

BUNDLE = f"real{args.symbols}x{args.bars}"


def _naive(idx):
    return idx.tz_localize(None) if getattr(idx, "tz", None) is not None else idx


def make_bundle(environ, asset_db_writer, minute_bar_writer, daily_bar_writer,
                adjustment_writer, calendar, start_session, end_session, cache,
                show_progress, output_dir):
    sessions = _naive(calendar.sessions_in_range(start_session, end_session))

    def frames():
        for i, sym in enumerate(names):
            frame = pd.DataFrame(
                {
                    "open": cols["open"][i],
                    "high": cols["high"][i],
                    "low": cols["low"][i],
                    "close": cols["close"][i],
                    "volume": cols["volume"][i],
                },
                index=index,
            ).reindex(_naive(sessions))
            if frame.isna().any().any():
                raise ValueError(f"Zipline calendar has unavailable bars for {sym}; refusing price fill")
            yield i, frame

    metadata = pd.DataFrame(
        {
            "symbol": names,
            "start_date": [sessions[0]] * len(names),
            "end_date": [sessions[-1]] * len(names),
            "auto_close_date": [sessions[-1] + pd.Timedelta(days=1)] * len(names),
            "exchange": ["NYSE"] * len(names),
        },
        index=range(len(names)),
    )
    asset_db_writer.write(equities=metadata)
    daily_bar_writer.write(frames(), show_progress=False)
    adjustment_writer.write()


bundles.register(BUNDLE, make_bundle, calendar_name="NYSE",
                 start_session=index[0], end_session=index[-1])
if ZIPLINE_ROOT.exists():
    shutil.rmtree(ZIPLINE_ROOT, ignore_errors=True)
bundles.ingest(BUNDLE, os.environ, show_progress=False)

want_by_symbol = {names[s]: want[s] for s in range(args.symbols)}
state = {"cursor": {}, "long": {}, "orders": 0}
feed = common.MtfFeed(ts, cols) if args.scenario == "mtf" else None


def initialize(context):
    set_slippage(slippage.FixedSlippage(spread=0.0))
    set_commission(commission.PerShare(cost=0.0, min_trade_cost=0.0))
    context.assets = [bundles.load(BUNDLE, os.environ).asset_finder.lookup_symbol(s, None)
                      for s in names]


def handle_data(context, data):
    for asset in context.assets:
        sym = asset.symbol
        i = state["cursor"].get(sym, 0)
        state["cursor"][sym] = i + 1
        if i >= args.bars:
            continue
        if args.scenario == "mtf":
            k = names.index(sym)
            target = feed.target(k, i)
            delta = target - state["long"].get(sym, 0)
            if delta:
                state["long"][sym] = target
                state["orders"] += 1
                order(asset, delta)
            continue
        if args.scenario == "resting":
            state["orders"] += 1
            k = names.index(sym)
            order(asset, common.QTY, style=LimitOrder(round(cols["close"][k][i] * common.REST_FRACTION, 2)))
            continue
        target = bool(want_by_symbol[sym][i])
        if target == state["long"].get(sym, False):
            continue
        state["long"][sym] = target
        state["orders"] += 1
        order(asset, common.QTY if target else -common.QTY)


started = perf_counter()
perf = run_algorithm(
    start=index[0],
    end=index[-1],
    initialize=initialize,
    handle_data=handle_data,
    capital_base=common.INITIAL_CASH,
    data_frequency="daily",
    bundle=BUNDLE,
)
elapsed = perf_counter() - started
pnl = float(perf["portfolio_value"].iloc[-1]) - common.INITIAL_CASH
fills = int(perf["transactions"].apply(len).sum())

common.emit(
    "zipline-reloaded",
    elapsed,
    state["orders"],
    args.symbols,
    args.bars,
    "event-driven; bundle ingested outside the timer; an order placed on day T fills at "
    "day T+1's close",
    expected=common.expected_for(args.scenario, ts, cols, want),
    pnl=pnl,
    convention=common.NEXT_CLOSE,
    fills=fills,
)
