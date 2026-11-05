"""Event-driven backtest engine."""

from __future__ import annotations

from array import array
from datetime import datetime, time, timedelta
from typing import Any, Literal, Mapping, cast

import numpy as np
import polars as pl

from .broker import BrokerSimulator
from .analyzers import EventAnalyzer
from .feed import ColumnarBarFeed, MarketSlice, market_slices_from_bars
from .record_sink import EventRecordSink
from .manifest import _BarFingerprintAccumulator, _manifest_with_fingerprint, build_run_manifest, strategy_configuration
from ..contracts import Bar, FillPrice, Order, OrderRecord
from .result import BacktestResult, EQUITY_COLUMNS, _empty_equity_curve_pl
from .dispatch import StrategyHooks, bind_strategy
from .strategy import Strategy, validate_strategy_hooks


SnapshotFrequency = Literal["bar", "session", "final", "none"]
ResultRetention = Literal["full", "summary"]

_EQUITY_VALUE_COLUMNS = EQUITY_COLUMNS[1:]
# The five the ledger measures; gross, net and leverage follow from them.
_EQUITY_CORE_COLUMNS = EQUITY_COLUMNS[1:6]


class _EquityAccumulator:
    """Columnar store for per-bar equity snapshots.

    Snapshots used to be kept as a list of row tuples and turned into a frame
    with `orient="row"`. That retained a tuple plus eight boxed floats per bar —
    roughly 310 bytes where the values are 64 — and then paid a row-to-column
    transpose at the end. Appending into `array("d")` keeps the doubles unboxed
    and lets Polars take each column directly; the row tuple the broker hands
    back becomes garbage immediately instead of being held for the whole run.

    Timestamps stay a plain list because the entries are references to datetimes
    the bars already own, so they cost a pointer each.

    Gross, net and leverage are not stored when the engine is producing rows
    itself: they are pure functions of long, short and equity, so five appends
    per bar and one vectorized pass at the end replace eight appends per bar. A
    broker that overrides `equity_snapshot_row` gets the full nine-column path,
    because then those three mean whatever that override says they mean.
    """

    __slots__ = ("timestamps", "_columns", "_appenders", "_derived", "append")

    def __init__(self, *, derived: bool = False) -> None:
        self.timestamps: list[datetime] = []
        self._derived = derived
        stored = _EQUITY_CORE_COLUMNS if derived else _EQUITY_VALUE_COLUMNS
        self._columns = tuple(array("d") for _ in stored)
        # Pre-bound append methods: looking up .append on each array per bar
        # allocates a bound-method object every call.
        self._appenders = (self.timestamps.append, *(column.append for column in self._columns))
        # Bound once instead of branching per bar. The two bodies are unrolled
        # rather than looped: the arity is fixed and a `for` over it costs more
        # than the calls it makes.
        self.append = self._append_core if derived else self._append_full

    def __len__(self) -> int:
        return len(self.timestamps)

    def _append_core(self, row: tuple) -> None:
        ts, eq, cash, pend, long_, short = self._appenders
        ts(row[0])
        eq(row[1])
        cash(row[2])
        pend(row[3])
        long_(row[4])
        short(row[5])

    def _append_full(self, row: tuple) -> None:
        ts, eq, cash, pend, long_, short, gross, net, lev = self._appenders
        ts(row[0])
        eq(row[1])
        cash(row[2])
        pend(row[3])
        long_(row[4])
        short(row[5])
        gross(row[6])
        net(row[7])
        lev(row[8])

    def to_frame(self) -> pl.DataFrame:
        stored = _EQUITY_CORE_COLUMNS if self._derived else _EQUITY_VALUE_COLUMNS
        columns = {
            "timestamp": self.timestamps,
            **{
                name: pl.Series(name, self._columns[index], dtype=pl.Float64)
                for index, name in enumerate(stored)
            },
        }
        if self._derived:
            # numpy over the raw doubles rather than Polars expressions: the
            # expression path costs more in fixed overhead than it saves on a
            # short curve, and a parameter sweep is thousands of short curves.
            long_exposure = np.frombuffer(self._columns[3], dtype=np.float64)
            short_exposure = np.frombuffer(self._columns[4], dtype=np.float64)
            equity = np.frombuffer(self._columns[0], dtype=np.float64)
            gross = long_exposure - short_exposure
            columns["gross_exposure"] = pl.Series("gross_exposure", gross, dtype=pl.Float64)
            columns["net_exposure"] = pl.Series(
                "net_exposure", long_exposure + short_exposure, dtype=pl.Float64
            )
            leverage = np.zeros_like(gross)
            np.divide(gross, equity, out=leverage, where=equity != 0.0)
            columns["leverage"] = pl.Series("leverage", leverage, dtype=pl.Float64)
        return pl.DataFrame(columns).select(EQUITY_COLUMNS)


_ONE_DAY = timedelta(days=1)
_MIDNIGHT = time.min


class BacktestEngine:
    """Single-use event loop. Construct a fresh engine and broker for each run."""

    result_type = BacktestResult

    def __init__(
        self,
        *,
        broker: BrokerSimulator | None = None,
        trade_on_close: bool = False,
        analyzers: list[EventAnalyzer] | None = None,
        snapshot_frequency: SnapshotFrequency = "bar",
        result_retention: ResultRetention = "full",
        record_sink: EventRecordSink | None = None,
        run_metadata: Mapping[str, Any] | None = None,
        build_manifest: bool = False,
        data_validation: Literal["preflight", "stream"] = "preflight",
        opening_times: Mapping[datetime, datetime] | None = None,
    ) -> None:
        if snapshot_frequency not in {"bar", "session", "final", "none"}:
            raise ValueError("snapshot_frequency must be one of: bar, session, final, none")
        if result_retention not in {"full", "summary"}:
            raise ValueError("result_retention must be one of: full, summary")
        if data_validation not in {"preflight", "stream"}:
            raise ValueError("data_validation must be preflight or stream")
        self.data_validation = data_validation
        # An engine-built broker fills at the price its timing allows.
        self.broker = broker or BrokerSimulator(fill_price=FillPrice.CLOSE if trade_on_close else FillPrice.OPEN)
        self.trade_on_close = trade_on_close
        if trade_on_close and self.broker.fill_price is FillPrice.OPEN:
            # The decision already saw this bar's close; filling at its open is look-ahead.
            raise ValueError("trade_on_close=True needs a non-open fill_price, e.g. FillPrice.CLOSE")
        # Explicit close -> open instants enable opening-print execution. Keep
        # the original bars for close callbacks, marks, and data fingerprints.
        self.opening_times = dict(opening_times) if opening_times is not None else None
        if opening_times is not None:
            if trade_on_close or self.broker.fill_price is not FillPrice.OPEN:
                raise ValueError("opening_times requires open-price execution and trade_on_close=False")
            for close, opening in opening_times.items():
                if close.tzinfo is None or opening.tzinfo is None or opening >= close:
                    raise ValueError("opening_times requires aware instants with open before close")
        self.analyzers = analyzers or []
        self.snapshot_frequency = snapshot_frequency
        self.result_retention = result_retention
        self.record_sink = record_sink
        self.run_metadata = dict(run_metadata or {})
        self.build_manifest = build_manifest
        self._has_run = False
        self._notified_order_records = 0
        self._order_notifier = None
        self._notify_scan = True

    def run(
        self,
        bars: list[Bar] | ColumnarBarFeed,
        strategy: StrategyHooks,
        *,
        assume_sorted: bool = False,
    ) -> BacktestResult:
        if self._has_run:
            raise RuntimeError(
                "BacktestEngine is single-use, including failed runs; construct a fresh "
                "engine and broker per run, or use engine_factory for optimization."
            )
        self._has_run = True
        strategy_config = strategy_configuration(strategy) if self.build_manifest else None
        runtime_retain_records = self.result_retention == "full" or any(
            getattr(analyzer, "requires_records", True) for analyzer in self.analyzers)
        before_analyzers = [a for a in self.analyzers
                            if getattr(a.before_bar, "__func__", None) is not EventAnalyzer.before_bar]
        after_analyzers = [a for a in self.analyzers
                           if getattr(a.after_bar, "__func__", None) is not EventAnalyzer.after_bar]
        slice_analyzers = [a for a in self.analyzers
                           if getattr(a.after_slice, "__func__", None) is not EventAnalyzer.after_slice]
        broker = self.broker
        broker.set_record_retention(runtime_retain_records)
        broker.set_record_sink(self.record_sink)
        self._notified_order_records = len(broker.order_records)
        dispatch = bind_strategy(strategy, broker)
        self._order_notifier = dispatch.notify_order
        # Order records only need per-step scanning when a notify hook actually
        # consumes them. Without a hook, tell the order manager not to accumulate
        # pending records at all, so summary runs skip the drain-and-drop work.
        has_notifier = self._order_notifier is not None
        broker.order_manager.track_pending_records = has_notifier
        self._notify_scan = has_notifier
        initial_equity = broker.state.equity()
        for analyzer in self.analyzers:
            analyzer.on_start(broker=broker, strategy=strategy)
        validate_strategy_hooks(strategy)
        strategy.initialize()
        builtin_snapshot = (
            getattr(broker.equity_snapshot_row, "__func__", None)
            is BrokerSimulator.equity_snapshot_row
        )
        # The core producer skips the three derived columns; they are computed
        # once over the whole curve instead of on every bar.
        snapshot_row = (
            broker.ledger.equity_snapshot_core if builtin_snapshot else broker.equity_snapshot_row
        )
        snapshot_rows = _EquityAccumulator(derived=builtin_snapshot)
        day_rollover: datetime | None = None
        pending_row = None

        feed = market_slices_from_bars(bars, assume_sorted=assume_sorted)
        streaming = self.data_validation == "stream"
        if not streaming:
            _reject_double_adjustment(feed, broker)
        fingerprint = _BarFingerprintAccumulator() if streaming and self.build_manifest else None
        run_manifest = None
        if self.build_manifest:
            run_manifest = build_run_manifest(
                (),
                broker=broker,
                strategy=strategy,
                trade_on_close=self.trade_on_close,
                snapshot_frequency=self.snapshot_frequency,
                result_retention=self.result_retention,
                metadata=self.run_metadata,
                data_fingerprint=fingerprint.finish() if fingerprint is not None else feed.data_fingerprint(),
                strategy_config=strategy_config,
                opening_times=self.opening_times,
            )

        trade_on_close = self.trade_on_close
        snapshot_frequency = self.snapshot_frequency
        take_bar_snapshots = snapshot_frequency == "bar"
        keep_pending_snapshot = snapshot_frequency in {"session", "final"}
        flush_session_snapshot = snapshot_frequency == "session"
        # Bind one level deeper when the broker does not override these
        # per-bar hooks; the wrapper frame is measurable at millions of bars.
        mark = broker.mark
        broker_mark = (
            broker.ledger.mark if getattr(mark, "__func__", None) is BrokerSimulator.mark else mark
        )
        notify_scan = self._notify_scan
        # With an empty book the opening fill pass only marks each bar and creates
        # no order records. Skip it (as the close path does) unless a broker
        # override could observe the pass itself.
        order_manager = broker.order_manager
        idle_open_pass_is_mark = (
            self.opening_times is None
            and getattr(broker.process_bar, "__func__", None) is BrokerSimulator.process_bar
            and getattr(broker.expire_orders, "__func__", None) is BrokerSimulator.expire_orders
        )
        on_bar = dispatch.on_bar
        on_bars = dispatch.on_bars
        drain_pending = dispatch.drain_pending
        on_before_bars = dispatch.on_before_bars
        before_trading_day = dispatch.before_trading_day
        set_market_slice = dispatch.set_market_slice
        last_timestamp: datetime | None = None

        single_bars = feed.single_bars
        engine_slices = single_bars if single_bars is not None else feed
        for market_slice in engine_slices:
            if single_bars is not None:
                bar = cast(Bar, market_slice)
                timestamp = bar.timestamp
                market_bars = [bar]
            else:
                grouped = cast("MarketSlice", market_slice)
                timestamp = grouped.timestamp
                market_bars = list(grouped.bars)
            if streaming and broker.corporate_actions:
                _reject_adjusted_bars(market_bars)
            if fingerprint is not None:
                fingerprint.update(market_bars)
            opening = timestamp
            opening_bars = market_bars
            if self.opening_times is not None:
                if timestamp not in self.opening_times:
                    raise ValueError(f"Missing opening time for {timestamp}")
                opening = self.opening_times[timestamp]
                if last_timestamp is not None and opening < last_timestamp:
                    raise ValueError("Session opening precedes the previous bar close")
                opening_bars = [bar._replace(timestamp=opening, high=bar.open, low=bar.open,
                                             close=bar.open, adj_close=None, vwap=None,
                                             volume=None, open_interest=None)
                                for bar in market_bars]
            last_timestamp = timestamp
            if set_market_slice is not None:
                set_market_slice(opening_bars)
            # Bars are sorted by non-decreasing timestamp, so the trading day can
            # only change when the timestamp reaches the next midnight. Caching
            # that threshold avoids a datetime.date() call on every bar.
            if day_rollover is None or opening >= day_rollover:
                trading_day = opening.date()
                day_rollover = datetime.combine(trading_day + _ONE_DAY, _MIDNIGHT, opening.tzinfo)
                if flush_session_snapshot and pending_row is not None:
                    snapshot_rows.append(pending_row)
                    pending_row = None
                broker.before_trading_day(opening)
                if before_trading_day is not None:
                    before_trading_day(trading_day, opening_bars)

            if not trade_on_close:
                before_orders = (
                    list(on_before_bars(opening_bars)) if on_before_bars is not None else []
                )
                if before_orders or notify_scan:
                    self._submit_strategy_orders(before_orders, strategy)
                if idle_open_pass_is_mark and order_manager.open_order_count == 0:
                    for bar in market_bars:
                        broker_mark(bar)
                else:
                    # Session volume remains the broker's capacity approximation;
                    # it is not observable by the strategy at the opening instant.
                    fill_bars = ([bar._replace(volume=original.volume)
                                  for bar, original in zip(opening_bars, market_bars, strict=True)]
                                 if self.opening_times is not None else market_bars)
                    self._process_market_bars(fill_bars, strategy)
                    if self.opening_times is not None:
                        for bar in market_bars:
                            broker_mark(bar)
            else:
                for bar in market_bars:
                    broker_mark(bar)

            if set_market_slice is not None and self.opening_times is not None:
                set_market_slice(market_bars)
            if before_analyzers:
                for bar in market_bars:
                    for analyzer in before_analyzers:
                        analyzer.before_bar(bar, broker=broker, strategy=strategy)

            if on_bars is not None:
                orders = list(on_bars(market_bars))
                if drain_pending is not None:
                    pending = drain_pending()
                    if pending:
                        orders.extend(pending)
            else:
                orders = []
                if on_bar is not None:
                    for bar in market_bars:
                        returned = on_bar(bar)
                        if returned:
                            orders.extend(returned)
            if orders or notify_scan:
                self._submit_strategy_orders(orders, strategy)

            if trade_on_close:
                self._process_market_bars(market_bars, strategy, mark_bars=False)

            if take_bar_snapshots:
                snapshot_rows.append(snapshot_row(timestamp))
            elif keep_pending_snapshot:
                pending_row = snapshot_row(timestamp)
            if after_analyzers:
                for analyzer in after_analyzers:
                    for bar in market_bars:
                        analyzer.after_bar(bar, broker=broker, strategy=strategy)
            for analyzer in slice_analyzers:
                analyzer.after_slice(timestamp, broker=broker, strategy=strategy)

        if fingerprint is not None and run_manifest is not None:
            run_manifest = _manifest_with_fingerprint(run_manifest, fingerprint.finish())

        # Settle the final session's futures P&L. Settlement normally runs at a
        # session boundary, and the last bar has no boundary after it, so
        # without this the run ends with the closing day's gain still unsettled
        # and `broker.state.cash` short by it. Equity is unaffected either way —
        # settlement moves value from unsettled variation margin into cash — so
        # the per-bar rows already recorded stay correct as of their own instant.
        settle_final = getattr(broker, "settle_variation_margin", None)
        if settle_final is not None and last_timestamp is not None:
            settle_final(last_timestamp)

        if keep_pending_snapshot and pending_row is not None:
            snapshot_rows.append(pending_row)
        strategy.finalize()
        self._notify_new_order_records()
        if isinstance(strategy, Strategy):
            strategy._finish_algo_orders()
        self._clear_strategy_market_slice(strategy)
        analyzer_results = {
            analyzer.result_name: analyzer.on_finish(broker=broker, strategy=strategy)
            for analyzer in self.analyzers
        }
        equity_curve = (
            snapshot_rows.to_frame() if snapshot_rows.timestamps else _empty_equity_curve_pl()
        )
        fills = self.broker.fills
        order_records = self.broker.order_records
        open_orders = self.broker.open_orders
        fill_count = self.broker.fill_count
        submitted_order_count = self.broker.submitted_order_count
        order_record_count = self.broker.order_record_count
        open_order_count = len(open_orders)
        records_retained = self.result_retention == "full"
        result = self.result_type(
            equity_curve=equity_curve,
            fills=list(fills) if records_retained else [],
            orders=list(order_records) if records_retained else [],
            open_orders=list(open_orders) if records_retained else None,
            positions=list(self.broker.state.positions.values()) if records_retained else None,
            analyzer_results=analyzer_results,
            algo_orders=strategy.algo_order_progress() if isinstance(strategy, Strategy) else [],
            fill_count=fill_count,
            submitted_order_count=submitted_order_count,
            order_record_count=order_record_count,
            open_order_count=open_order_count,
            total_commission=self.broker.ledger.total_commission,
            total_slippage=self.broker.ledger.total_slippage,
            records_retained=records_retained,
            initial_equity=initial_equity,
            contract_multipliers=self.broker.state.contract_multipliers,
            run_manifest=run_manifest,
            corporate_action_events=list(self.broker.corporate_action_events or []) if records_retained else [],
        )
        if self.record_sink is not None:
            self.record_sink.write_final_tables(
                equity_curve=equity_curve,
                open_orders=open_orders,
                positions=list(self.broker.state.positions.values()),
            )
            self.record_sink.close()
            self.broker.set_record_sink(None)
        return result

    def _process_market_bars(
        self, bars: list[Bar], strategy: StrategyHooks, *, mark_bars: bool = True
    ) -> None:
        if not mark_bars and self.broker.order_manager.open_order_count == 0:
            return
        notify_scan = self._notify_scan
        # Hoisted: this runs once per timestamp, and a single-symbol run has one
        # timestamp per observation, so every repeated attribute walk here is
        # paid on every bar of the whole backtest.
        broker = self.broker
        analyzers = self.analyzers
        if bars:
            broker.expire_orders(bars[0].timestamp)
            if notify_scan:
                self._notify_new_order_records()
        for bar in bars:
            fills = broker.process_bar(bar, expire_orders=False, mark=False)
            if mark_bars or fills:
                # Marking happens after the fill, never before it. With
                # trade_on_close=False this bar fills at its open, so cash and
                # risk checks must see equity as of the previous close rather
                # than a close that has not happened yet. A fill also stamps the
                # position with the (slippage-adjusted) fill price, so snapshots
                # still need this bar's close mark either way.
                broker.mark(bar)
            if notify_scan:
                self._notify_new_order_records()
            if analyzers:
                for fill in fills:
                    for analyzer in analyzers:
                        analyzer.on_fill(fill, broker=broker, strategy=strategy)

    def _submit_strategy_orders(self, orders: list[Order], strategy: StrategyHooks) -> None:
        notify_scan = self._notify_scan
        if notify_scan:
            self._notify_new_order_records()
        for order in orders:
            record = self.broker.submit_order(order)
            if notify_scan:
                self._notify_new_order_records()
            if self.analyzers:
                for analyzer in self.analyzers:
                    analyzer.on_order(order, record, broker=self.broker, strategy=strategy)

    @staticmethod
    def _clear_strategy_market_slice(strategy: StrategyHooks) -> None:
        clearer = getattr(strategy, "_clear_market_slice", None)
        if clearer is not None:
            clearer()

    def _notify_new_order_records(self) -> None:
        if not self.broker.order_manager.retain_records:
            records = self.broker.drain_order_records()
            if self._order_notifier is None:
                self._notified_order_records += len(records)
                return
            for record in records:
                self._notify_order_record(record)
            return
        records = self.broker.order_records
        if self._order_notifier is None:
            self._notified_order_records = len(records)
            return
        while self._notified_order_records < len(records):
            self._notify_order_record(records[self._notified_order_records])

    def _notify_order_record(self, record: OrderRecord) -> None:
        if self._order_notifier is not None:
            self._order_notifier(record)
        self._notified_order_records += 1


def _reject_double_adjustment(feed: ColumnarBarFeed, broker: BrokerSimulator) -> None:
    """Refuse to apply corporate actions on top of already-adjusted prices.

    ``adj_close`` has splits and dividends baked in, and the ledger marks on it,
    so replaying the same actions would count them twice and silently corrupt
    equity. Only scans when the caller opted into corporate actions, and stops at
    the first offending bar.
    """
    actions = getattr(broker, "corporate_actions", None)
    if not actions:
        return
    for market_slice in feed:
        _reject_adjusted_bars(market_slice.bars)


def _reject_adjusted_bars(bars) -> None:
    for bar in bars:
        if bar.adj_close is not None:
            raise ValueError(
                    f"bar {bar.symbol} carries adj_close while the broker has corporate "
                    "actions configured; corporate actions apply to unadjusted prices only. "
                    "Feed raw close without adj_close, or drop the corporate actions."
                )
