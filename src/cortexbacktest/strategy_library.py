"""Causal, bounded-history examples for research and workload testing.

These are explicit long-only variants of familiar strategy families, not
claims of profitability. Periods count observed bars, not trading days. Run
with trade_on_close=False and OPEN fills to execute signals on the next bar.
See docs/benchmarks.md for source links and exact rules.
"""
from __future__ import annotations

from collections import deque
from collections.abc import Callable
from heapq import nlargest
from math import fsum, isfinite, sqrt

from .contracts import Bar, Order, Side
from .event.strategy import Strategy


def _period(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 2:
        raise ValueError(f'{name} must be an integer >= 2')


class _RollingMoments:
    """Sliding Welford updates with periodic rebasing; amortized O(1)."""
    def __init__(self, period: int):
        self.period = period
        self.values: deque[float] = deque()
        self.mean = self.m2 = 0.0
        self.updates = 0

    def update(self, value: float) -> None:
        n = len(self.values)
        if n == self.period:
            old = self.values.popleft()
            mean = self.mean
            self.mean += (mean - old) / (n - 1)
            self.m2 -= (old - mean) * (old - self.mean)
            n -= 1
        self.values.append(value)
        delta = value - self.mean
        self.mean += delta / (n + 1)
        self.m2 += delta * (value - self.mean)
        self.updates += 1
        if self.updates % self.period == 0:
            self.mean = fsum(self.values) / len(self.values)
            self.m2 = fsum((x - self.mean) ** 2 for x in self.values)

    @property
    def ready(self) -> bool:
        return len(self.values) == self.period

    @property
    def std(self) -> float:
        return sqrt(max(0.0, self.m2) / len(self.values)) if self.values else 0.0


class _TargetStrategy(Strategy):
    def __init__(self, quantity: float = 1.0):
        super().__init__()
        if not isfinite(quantity) or quantity <= 0:
            raise ValueError('quantity must be finite and positive')
        self.quantity = quantity
        self._targets: dict[str, float] = {}
        # Optional timing filter: while it returns False for a bar, that symbol is held flat.
        self.gate: Callable[[Bar], bool] | None = None

    def on_bars(self, bars: list[Bar]) -> list[Order]:
        self.observe_targets(bars)
        return self._rebalance(bars)

    def observe_targets(self, bars: list[Bar]) -> dict[str, float]:
        """Advance the model on one slice without a broker; the shadow run calls this directly."""
        raise NotImplementedError

    @staticmethod
    def position_delta(target: float, current: float) -> float:
        delta = target - current
        return delta if abs(delta) > 1e-10 else 0.0

    def _rebalance(self, bars: list[Bar]) -> list[Order]:
        orders = []
        manager = self.broker.order_manager
        for bar in bars:
            target = self._targets.get(bar.symbol)
            if target is None or bar.halted:
                continue
            if target and self.gate is not None and not self.gate(bar):
                target = 0.0
            position = self.broker.state.positions.get(bar.symbol)
            current = position.quantity if position is not None else 0.0
            pending = manager.open_orders_for_symbol(bar.symbol)
            expected = current + sum(o.quantity if o.side == Side.BUY else -o.quantity for o in pending)
            if abs(target - expected) <= 1e-10:
                continue
            # Signal reversal cancels remaining quantity, including partial fills.
            for order in pending:
                if order.id is not None:
                    self.broker.cancel_order(order.id, reason='strategy target changed')
            delta = self.position_delta(target, current)
            if delta:
                orders.append(Order(bar.symbol, bar.timestamp,
                                    Side.BUY if delta > 0 else Side.SELL, abs(delta)))
        # Sell requests precede buys, providing deterministic cash release order.
        return sorted(orders, key=lambda order: order.side == Side.BUY)


class BuyHold(_TargetStrategy):
    """Target a fixed quantity of each observed symbol, retrying rejected orders."""
    def observe_targets(self, bars: list[Bar]) -> dict[str, float]:
        for bar in bars:
            self._targets[bar.symbol] = self.quantity
        return dict(self._targets)


class SmaCross(_TargetStrategy):
    """Long when fast SMA > slow SMA after both windows are warm; otherwise flat."""
    def __init__(self, fast: int = 15, slow: int = 30, quantity: float = 1.0):
        super().__init__(quantity)
        _period(fast, 'fast')
        _period(slow, 'slow')
        if fast >= slow:
            raise ValueError('fast must be below slow')
        self.fast, self.slow = fast, slow
        self._windows: dict[str, tuple[_RollingMoments, _RollingMoments]] = {}

    def observe_targets(self, bars: list[Bar]) -> dict[str, float]:
        return self.observe_closes([(bar.symbol, bar.return_close) for bar in bars if not bar.halted])

    def observe_closes(self, prices: list[tuple[str, float]]) -> dict[str, float]:
        """Update model state only; historical warmup never generates orders."""
        for symbol, close in prices:
            windows = self._windows.get(symbol)
            if windows is None:
                windows = self._windows[symbol] = (_RollingMoments(self.fast), _RollingMoments(self.slow))
            fast, slow = windows
            fast.update(close)
            slow.update(close)
            if slow.ready:
                self._targets[symbol] = self.quantity if fast.mean > slow.mean else 0.0
        return dict(self._targets)


class RsiReversion(_TargetStrategy):
    """Wilder RSI: enter below lower; exit above upper; retain target in between."""
    def __init__(self, period: int = 14, lower: float = 30, upper: float = 70, quantity: float = 1.0):
        super().__init__(quantity)
        _period(period, 'period')
        if not 0 <= lower < upper <= 100:
            raise ValueError('require 0 <= lower < upper <= 100')
        self.period, self.lower, self.upper = period, lower, upper
        # previous close, number of deltas, mean gain, mean loss
        self._state: dict[str, tuple[float, int, float, float]] = {}

    def observe_targets(self, bars: list[Bar]) -> dict[str, float]:
        for bar in bars:
            if bar.halted:
                continue
            previous = self._state.get(bar.symbol)
            if previous is None:
                self._state[bar.symbol] = (bar.return_close, 0, 0.0, 0.0)
                continue
            close, count, gain, loss = previous
            change = bar.return_close - close
            count += 1
            divisor = min(count, self.period)
            gain += (max(change, 0.0) - gain) / divisor
            loss += (max(-change, 0.0) - loss) / divisor
            self._state[bar.symbol] = (bar.return_close, count, gain, loss)
            if count >= self.period:
                rsi = 50.0 if gain == loss == 0 else 100.0 if loss == 0 else 100 - 100 / (1 + gain / loss)
                if rsi < self.lower:
                    self._targets[bar.symbol] = self.quantity
                elif rsi > self.upper:
                    self._targets[bar.symbol] = 0.0
        return dict(self._targets)


class BollingerReversion(_TargetStrategy):
    """Enter below mean-k*population_std; exit at/above mean, including current close."""
    def __init__(self, period: int = 20, deviations: float = 2.0, quantity: float = 1.0):
        super().__init__(quantity)
        _period(period, 'period')
        if not isfinite(deviations) or deviations <= 0:
            raise ValueError('deviations must be finite and positive')
        self.period, self.deviations = period, deviations
        self._windows: dict[str, _RollingMoments] = {}

    def observe_targets(self, bars: list[Bar]) -> dict[str, float]:
        for bar in bars:
            if bar.halted:
                continue
            window = self._windows.get(bar.symbol)
            if window is None:
                window = self._windows[bar.symbol] = _RollingMoments(self.period)
            window.update(bar.return_close)
            if window.ready:
                if bar.return_close < window.mean - self.deviations * window.std:
                    self._targets[bar.symbol] = self.quantity
                elif bar.return_close >= window.mean:
                    self._targets[bar.symbol] = 0.0
        return dict(self._targets)


class MomentumRotation(_TargetStrategy):
    """Every interval slices, hold top-k positive lookback returns at fixed size.

    Rank only warmed symbols present in the current slice. Absent positions
    receive a flat target and are closed when a later tradable bar arrives.
    This is fixed-share top-k exposure, not a claim of equal-weight allocation.
    """
    def __init__(self, lookback: int = 60, top_k: int = 10, interval: int = 20, quantity: float = 1.0):
        super().__init__(quantity)
        _period(lookback, 'lookback')
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in (top_k, interval)):
            raise ValueError('top_k and interval must be positive integers')
        self.lookback, self.top_k, self.interval = lookback, top_k, interval
        self._windows: dict[str, deque[float]] = {}
        self._slices = 0

    def observe_targets(self, bars: list[Bar]) -> dict[str, float]:
        self._slices += 1
        rebalance = self._slices % self.interval == 0
        scores = []
        for bar in bars:
            if bar.halted:
                continue
            window = self._windows.get(bar.symbol)
            if window is None:
                window = self._windows[bar.symbol] = deque(maxlen=self.lookback + 1)
            window.append(bar.return_close)
            if rebalance and len(window) == self.lookback + 1:
                score = bar.return_close / window[0] - 1
                if score > 0:
                    scores.append((score, bar.symbol))
        if rebalance:
            selected = {symbol for _, symbol in nlargest(self.top_k, scores)}
            self._targets = dict.fromkeys(self._windows, 0.0)
            self._targets.update(dict.fromkeys(selected, self.quantity))
        return dict(self._targets)


STRATEGIES = {'buy_hold': BuyHold, 'sma_cross': SmaCross, 'rsi_reversion': RsiReversion,
              'bollinger_reversion': BollingerReversion, 'momentum_rotation': MomentumRotation}

__all__ = ["BuyHold", "SmaCross", "RsiReversion", "BollingerReversion", "MomentumRotation", "STRATEGIES"]
