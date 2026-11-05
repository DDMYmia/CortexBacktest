from datetime import UTC, datetime, timedelta
from math import fsum

import numpy as np
import polars as pl
import pytest

from cortexbacktest import BacktestEngine, BrokerSimulator, ColumnarBarFeed, ParquetBarFeed
from cortexbacktest.strategy_library import (
    STRATEGIES, SmaCross, RsiReversion, BollingerReversion, MomentumRotation,
    _RollingMoments, _TargetStrategy,
)


def prices(minutes=180, symbols=4):
    rng = np.random.default_rng(72)
    values = 100 * np.exp(np.cumsum(rng.normal(0, .035, (minutes, symbols)), axis=0))
    return pl.DataFrame({
        'date': [datetime(2024, 1, 1, tzinfo=UTC) + timedelta(minutes=i)
                 for i in range(minutes) for _ in range(symbols)],
        'symbol': [f'S{i}' for i in range(symbols)] * minutes,
        'open': values.flatten(), 'high': values.flatten() * 1.01,
        'low': values.flatten() * .99, 'close': values.flatten(),
        'volume': [12.] * (minutes * symbols),
    })


class Reference(_TargetStrategy):
    """Slow full-history oracles, intentionally separate from rolling helpers."""
    def __init__(self, name, quantity=10):
        super().__init__(quantity)
        self.name = name
        self.history = {}
        self.slices = 0

    def on_bars(self, bars):
        self.slices += 1
        scores = []
        for bar in bars:
            h = self.history.setdefault(bar.symbol, [])
            h.append(bar.close)
            n = len(h)
            if self.name == 'buy_hold':
                self._targets[bar.symbol] = self.quantity
            elif self.name == 'sma_cross' and n >= 30:
                self._targets[bar.symbol] = self.quantity if np.mean(h[-15:]) > np.mean(h[-30:]) else 0.
            elif self.name == 'rsi_reversion' and n >= 15:
                changes = np.diff(h)
                gains = np.maximum(changes, 0)
                losses = np.maximum(-changes, 0)
                gain, loss = np.mean(gains[:14]), np.mean(losses[:14])
                for i in range(14, len(changes)):
                    gain = (gain * 13 + gains[i]) / 14
                    loss = (loss * 13 + losses[i]) / 14
                rsi = 50. if gain == loss == 0 else 100. if loss == 0 else 100 - 100 / (1 + gain / loss)
                if rsi < 30:
                    self._targets[bar.symbol] = self.quantity
                elif rsi > 70:
                    self._targets[bar.symbol] = 0.
            elif self.name == 'bollinger_reversion' and n >= 20:
                mean, std = np.mean(h[-20:]), np.std(h[-20:])
                if bar.close < mean - 2 * std:
                    self._targets[bar.symbol] = self.quantity
                elif bar.close >= mean:
                    self._targets[bar.symbol] = 0.
            elif self.name == 'momentum_rotation' and n >= 61:
                score = h[-1] / h[-61] - 1
                if score > 0:
                    scores.append((score, bar.symbol))
        if self.name == 'momentum_rotation' and self.slices % 20 == 0:
            selected = {s for _, s in sorted(scores, reverse=True)[:10]}
            self._targets = {s: self.quantity if s in selected else 0. for s in self.history}
        return self._rebalance(bars)


def run(feed, strategy, **kwargs):
    return BacktestEngine(broker=BrokerSimulator(initial_cash=100000, max_volume_share=.25,
            commission_bps=1, slippage_bps=1), trade_on_close=False, **kwargs).run(feed, strategy)


@pytest.mark.parametrize('name', STRATEGIES)
def test_strategies_match_full_history_oracle_and_parquet(tmp_path, name):
    data = prices()
    data.write_parquet(tmp_path / 'bars.parquet')
    cached = ColumnarBarFeed.from_polars(data)
    reference = run(cached, Reference(name))
    result = run(cached, STRATEGIES[name](quantity=10))
    streamed = run(ParquetBarFeed(tmp_path / 'bars.parquet', batch_rows=37),
                   STRATEGIES[name](quantity=10), build_manifest=True, data_validation='stream')
    assert result.fills == reference.fills == streamed.fills
    assert result.summary() == reference.summary() == streamed.summary()
    assert result.fill_count > 0
    assert all(position.quantity >= 0 for position in result.positions)
    # Signals based on a close may not fill at that same timestamp.
    submitted = {record.order.id: record.order.timestamp for record in result.orders}
    assert all(fill.timestamp > submitted[fill.order_id] for fill in result.fills)
    dates = set(data['date'].to_list())
    assert all(fill.timestamp in dates for fill in result.fills)


@pytest.mark.parametrize('name', STRATEGIES)
def test_future_data_cannot_change_prefix_fills(name):
    data = prices()
    prefix = run(ColumnarBarFeed.from_polars(data[:400]), STRATEGIES[name]())
    full = run(ColumnarBarFeed.from_polars(data), STRATEGIES[name]())
    cutoff = data['date'][399]
    assert prefix.fills == [f for f in full.fills if f.timestamp <= cutoff]


def test_rolling_moments_stable_and_bounded():
    window = _RollingMoments(20)
    values = [1e8 + np.sin(i) * .01 for i in range(400)]
    for i, value in enumerate(values):
        window.update(value)
        expected = values[max(0, i - 19):i + 1]
        assert len(window.values) <= 20
        assert window.mean == pytest.approx(fsum(expected) / len(expected), abs=1e-6)
        assert window.std == pytest.approx(np.std(expected), abs=1e-6)


@pytest.mark.parametrize('factory,kwargs', [
    (SmaCross, {'fast': 30, 'slow': 15}), (SmaCross, {'fast': True}),
    (RsiReversion, {'lower': 80, 'upper': 30}), (RsiReversion, {'period': 1}),
    (BollingerReversion, {'deviations': float('nan')}),
    (MomentumRotation, {'top_k': 0}), (MomentumRotation, {'interval': False}),
])
def test_bad_parameters(factory, kwargs):
    with pytest.raises(ValueError):
        factory(**kwargs)


def test_parquet_process_search_uses_paths_without_ipc(tmp_path, monkeypatch):
    import cortexbacktest.event.optimization as optimization
    import cortexbacktest.event.parallel as parallel
    data = prices(80, 3)
    path = tmp_path / 'search.parquet'
    data.write_parquet(path)
    # Workers must rebuild the feed with its filters, not just its paths.
    feed = ParquetBarFeed(path, batch_rows=37, symbols=['S0', 'S2'],
                          start=datetime(2024, 1, 1, 0, 10, tzinfo=UTC))
    def forbid(*args, **kwargs):
        raise AssertionError('Parquet search must not encode full IPC feed')
    monkeypatch.setattr(parallel, 'write_feed_ipc', forbid)
    serial = optimization.optimize_strategy(feed, SmaCross, fast=[3, 5, 7, 9], slow=[15],
                                            parallel_backend='serial')
    process = optimization.optimize_strategy(feed, SmaCross, fast=[3, 5, 7, 9], slow=[15],
                                             max_workers=2, parallel_backend='process')
    assert process.rows == serial.rows
    assert process.execution['backend'] == 'process'
    assert process.execution['feed_transport'] == 'parquet_paths'


def test_streaming_stats_match_retained_equity_and_sink(tmp_path):
    from cortexbacktest.event.analyzers import StreamingStatsAnalyzer
    from cortexbacktest.event.record_sink import EventRecordSink
    feed = ColumnarBarFeed.from_polars(prices())
    full = run(feed, SmaCross(quantity=10))
    broker = BrokerSimulator(initial_cash=100000, max_volume_share=.25, commission_bps=1, slippage_bps=1)
    result = BacktestEngine(broker=broker, trade_on_close=False, result_retention='summary',
        snapshot_frequency='final', analyzers=[StreamingStatsAnalyzer()],
        record_sink=EventRecordSink(tmp_path / 'records', format='parquet', buffer_rows=7)
    ).run(feed, SmaCross(quantity=10))
    stats = result.analyzer_results['streaming_stats']
    equity = np.r_[100000., full.performance.equity_curve()['equity'].to_numpy()]
    returns = equity[1:] / equity[:-1] - 1
    assert stats['slices'] == len(feed)
    assert stats['fills'] == full.fill_count
    assert stats['final_equity'] == equity[-1]
    assert stats['mean_return'] == pytest.approx(returns.mean())
    assert stats['return_variance'] == pytest.approx(returns.var())
    assert stats['max_drawdown'] == pytest.approx((equity / np.maximum.accumulate(equity) - 1).min())
    assert broker.fills == broker.order_records == []
    assert pl.read_parquet(tmp_path / 'records/fills.parquet').height == full.fill_count
    assert pl.read_parquet(tmp_path / 'records/orders.parquet').height == len(full.orders)


@pytest.mark.parametrize('name', STRATEGIES)
def test_library_is_available_through_dynamic_loader_and_config_api(name):
    from cortexbacktest.api import BacktestConfig, StrategySpec, run_backtest
    from cortexbacktest.event.strategy_loader import load_strategy
    cls = STRATEGIES[name]
    loaded = load_strategy(f'cortexbacktest.strategy_library:{cls.__name__}', kwargs={'quantity': 2})
    assert isinstance(loaded, cls)
    feed = ColumnarBarFeed.from_polars(prices(80, 2))
    result = run_backtest(feed, StrategySpec(cls, {'quantity': 2}),
                         config=BacktestConfig(data_validation='stream'))
    assert result.run_manifest['bar_count'] == 160


def test_rotation_selects_top_k_and_defers_missing_symbol_exit():
    from cortexbacktest import Bar
    from datetime import timedelta
    strategy = MomentumRotation(lookback=2, top_k=1, interval=1, quantity=1)
    strategy._bind_broker(BrokerSimulator(initial_cash=10000))
    start = datetime(2024, 1, 1, tzinfo=UTC)
    def bars(i, values):
        return [Bar(s, start + timedelta(minutes=i), p, p, p, p) for s, p in values.items()]
    strategy.on_bars(bars(0, {'A': 100., 'B': 100., 'C': 100.}))
    strategy.on_bars(bars(1, {'A': 110., 'B': 120., 'C': 90.}))
    orders = strategy.on_bars(bars(2, {'A': 120., 'B': 140., 'C': 80.}))
    assert [(o.symbol, o.quantity) for o in orders] == [('B', 1)]
    strategy.on_bars(bars(3, {'A': 150., 'C': 80.}))
    assert strategy._targets['A'] == 1
    assert strategy._targets['B'] == 0


def test_custom_history_analyzer_still_retains_records():
    from cortexbacktest.event.analyzers import EventAnalyzer
    class NeedsHistory(EventAnalyzer):
        def on_finish(self, *, broker, strategy):
            return {'fills': len(broker.fills), 'orders': len(broker.order_records)}
    feed = ColumnarBarFeed.from_polars(prices(80, 2))
    result = run(feed, SmaCross(), result_retention='summary', analyzers=[NeedsHistory()])
    assert result.analyzer_results['NeedsHistory']['fills'] == result.fill_count > 0
    assert result.analyzer_results['NeedsHistory']['orders'] > result.fill_count
