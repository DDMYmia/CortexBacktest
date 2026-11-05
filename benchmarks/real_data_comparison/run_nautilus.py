from time import perf_counter

import common

from nautilus_trader.backtest.engine import BacktestEngine, BacktestEngineConfig
from nautilus_trader.config import CacheConfig, LoggingConfig, RiskEngineConfig
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import AccountType, OmsType, OrderSide
from nautilus_trader.model.identifiers import Venue
from nautilus_trader.model.objects import Money, Price, Quantity
from nautilus_trader.test_kit.providers import TestInstrumentProvider
from nautilus_trader.trading.strategy import Strategy

args = common.cli()
names, ts, cols = common.load(args.dataset, args.symbols, args.bars)
want = common.signals(cols["close"])

engine = BacktestEngine(
    config=BacktestEngineConfig(
        trader_id="BACKTESTER-001",
        logging=LoggingConfig(bypass_logging=True),
        risk_engine=RiskEngineConfig(bypass=True),
        cache=CacheConfig(bar_capacity=10),
        run_analysis=False,
    )
)
engine.add_venue(
    venue=Venue("SIM"),
    oms_type=OmsType.NETTING,
    account_type=AccountType.MARGIN,
    base_currency=USD,
    starting_balances=[Money(1_000_000_000, USD)],
)

bar_types, want_by_id, index_of = [], {}, {}
feed = common.MtfFeed(ts, cols) if args.scenario == "mtf" else None
for i, sym in enumerate(names):
    inst = TestInstrumentProvider.equity(symbol=sym, venue="SIM")
    engine.add_instrument(inst)
    bt = BarType.from_str(f"{inst.id}-1-DAY-LAST-EXTERNAL")
    bar_types.append(bt)
    want_by_id[inst.id] = want[i]
    index_of[inst.id] = i

o, h, low, c, v = (cols[k] for k in ("open", "high", "low", "close", "volume"))
bars = []
for b in range(args.bars):
    stamp = int(ts[b])
    for i in range(args.symbols):
        bars.append(
            Bar(
                bar_types[i],
                Price.from_str(f"{o[i, b]:.2f}"),
                Price.from_str(f"{h[i, b]:.2f}"),
                Price.from_str(f"{low[i, b]:.2f}"),
                Price.from_str(f"{c[i, b]:.2f}"),
                Quantity.from_int(int(v[i, b])),
                stamp,
                stamp,
            )
        )
engine.add_data(bars)


class Sma(Strategy):
    def __init__(self, config=None):
        super().__init__(config)
        self.bar_types = []
        self.cursor = {}
        self.long = {}
        self.held = {}
        self.n_orders = 0

    def on_start(self):
        for bt in self.bar_types:
            self.subscribe_bars(bt)

    def on_bar(self, bar):
        iid = bar.bar_type.instrument_id
        i = self.cursor.get(iid, 0)
        self.cursor[iid] = i + 1
        if args.scenario == "mtf":
            target = feed.target(index_of[iid], i)
            delta = target - self.held.get(iid, 0)
            if delta:
                self.held[iid] = target
                self.n_orders += 1
                self.submit_order(self.order_factory.market(
                    instrument_id=iid, order_side=OrderSide.BUY if delta > 0 else OrderSide.SELL,
                    quantity=Quantity.from_int(abs(delta))))
            return
        if args.scenario == "resting":
            self.n_orders += 1
            price = Price.from_str(f"{float(bar.close) * common.REST_FRACTION:.2f}")
            self.submit_order(self.order_factory.limit(iid, OrderSide.BUY, Quantity.from_int(common.QTY), price))
            return
        target = bool(want_by_id[iid][i])
        if target == self.long.get(iid, False):
            return
        self.long[iid] = target
        self.n_orders += 1
        self.submit_order(
            self.order_factory.market(
                instrument_id=iid,
                order_side=OrderSide.BUY if target else OrderSide.SELL,
                quantity=Quantity.from_int(common.QTY),
            )
        )


strategy = Sma()
strategy.bar_types = bar_types
engine.add_strategy(strategy)

started = perf_counter()
engine.run()
elapsed = perf_counter() - started

# Equity from the engine's own fills, marked at each symbol's last close.
fills = engine.trader.generate_order_fills_report()
last_close = {f"{sym}.SIM": float(c[i, -1]) for i, sym in enumerate(names)}
pnl = 0.0
for row in fills.itertuples():
    qty = float(row.filled_qty) * (1 if str(row.side) == "BUY" else -1)
    pnl += -qty * float(row.avg_px) + qty * last_close[str(row.instrument_id)]

common.emit(
    "nautilus_trader",
    elapsed,
    strategy.n_orders,
    args.symbols,
    args.bars,
    "event-driven; full live-parity stack; logging/risk bypassed; MARGIN account "
    "(a CASH account aborts on AccountBalanceNegative past ~700 symbols)",
    expected=common.expected_for(args.scenario, ts, cols, want),
    pnl=pnl,
    convention=common.SAME_CLOSE,
    fills=len(fills),
)
