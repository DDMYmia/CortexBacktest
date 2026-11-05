"""A strategy that cannot place an order must be rejected, not run silently.

`Strategy` supplies a working default for every hook, so a subclass that
misspells one runs to completion and reports a flat equity curve. Nothing in the
result distinguishes "no edge" from "the hook was never called", which makes it
the most expensive kind of silent failure in a backtester.

`Strategy` is deliberately not an ABC — orders may come from `on_bar`,
`on_bars`, or `on_before_bars`, and the compatibility templates add their own on
top — so this check replaces an abstract method that could not have been
declared without breaking the templates.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import Mock

import pytest

from cortexbacktest import BacktestEngine, Bar, Order, Side, Strategy
from cortexbacktest.event.backtrader_compat import BacktraderStrategyMixin
from cortexbacktest.event.strategy import validate_strategy_hooks
from cortexbacktest.event.templates import BuyAndHoldStrategy
from cortexbacktest.event.vnpy_compat import PortfolioStrategyTemplate


def _bars(count: int = 3) -> list[Bar]:
    start = datetime(2024, 1, 1)
    return [
        Bar("AAA", start + timedelta(days=day), 100.0, 101.0, 99.0, 100.0, volume=1_000.0)
        for day in range(count)
    ]


class MisspelledHook(Strategy):
    """The failure this exists to catch: `on_bars` typed as `on_bar_s`."""

    def on_bar_s(self, bars):  # noqa: D102 - deliberately wrong name
        return [Order("AAA", bars[0].timestamp, Side.BUY, 1)]


class NoHooksAtAll(Strategy):
    pass


class UsesOnBar(Strategy):
    def on_bar(self, bar):
        return []


class UsesOnBars(Strategy):
    def on_bars(self, bars):
        return []


class UsesOnBeforeBars(Strategy):
    def on_before_bars(self, bars):
        return []


@pytest.mark.parametrize("strategy_class", [MisspelledHook, NoHooksAtAll])
def test_engine_rejects_a_strategy_that_can_never_trade(strategy_class):
    engine = BacktestEngine()
    with pytest.raises(TypeError, match="overrides none of"):
        engine.run(_bars(), strategy_class())


def test_the_error_names_the_hooks_and_the_class():
    with pytest.raises(TypeError) as caught:
        validate_strategy_hooks(MisspelledHook())
    message = str(caught.value)
    assert "MisspelledHook" in message
    for hook in ("on_bar", "on_bars", "on_before_bars"):
        assert hook in message
    assert "spelling" in message, "the message should point at the likely cause"


# Overriding any single hook is enough. The shipped templates dispatch to hooks of
# their own (`on_portfolio_bars`, Backtrader's `next`) by overriding `on_bars`, so
# they must pass without the user having to know about this check.
@pytest.mark.parametrize("strategy_class", [UsesOnBar, UsesOnBars, UsesOnBeforeBars, BuyAndHoldStrategy,
                                            PortfolioStrategyTemplate, BacktraderStrategyMixin])
def test_one_overridden_hook_satisfies_the_check(strategy_class):
    validate_strategy_hooks(strategy_class())


def test_strategy_is_not_abstract_and_stays_instantiable():
    # Dropping ABC is part of this change: a base that enforces nothing should
    # not claim to be abstract. Instantiating it directly must still work,
    # because the templates and compat layers subclass and delegate to it.
    assert Strategy() is not None


def test_noncallable_optional_hook_fails_at_binding():
    strategy = UsesOnBar()
    strategy.on_before_bars = 42
    with pytest.raises(TypeError, match="on_before_bars must be callable"):
        BacktestEngine().run(_bars(5), strategy)


def test_slotted_duck_strategy_and_instance_order_notifier():
    class Duck:
        __slots__ = ("calls",)

        def __init__(self):
            self.calls = 0

        def initialize(self):
            pass

        def on_bar(self, bar):
            self.calls += 1
            return []

        def finalize(self):
            pass

    strategy = Duck()
    BacktestEngine().run(_bars(5), strategy)
    assert strategy.calls == 5


def test_batch_only_duck_strategy_needs_no_on_bar():
    class BatchOnly:
        def initialize(self):
            self.calls = 0

        def on_bars(self, bars):
            self.calls += 1
            return []

        def finalize(self):
            pass

    strategy = BatchOnly()
    BacktestEngine().run(_bars(5), strategy)
    assert strategy.calls == 5


def test_instance_strategy_order_hook_is_a_valid_override():
    strategy = Strategy()
    strategy.on_bar = Mock(return_value=[])
    BacktestEngine().run(_bars(5), strategy)
    assert strategy.on_bar.call_count == 5
