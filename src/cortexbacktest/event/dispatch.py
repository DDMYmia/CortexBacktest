"""Typed strategy hook contract and one-time event-loop dispatch binding."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Callable, Iterable, Protocol, cast

from ..contracts import Bar, Order, OrderRecord
from .broker import BrokerSimulator
from .strategy import Strategy


class StrategyHooks(Protocol):
    """Required lifecycle for duck-typed strategies; extra hooks are optional."""

    def initialize(self) -> None: ...
    def finalize(self) -> None: ...


class BatchHook(Protocol):
    def __call__(self, bars: list[Bar]) -> Iterable[Order]: ...


@dataclass(frozen=True, slots=True)
class StrategyDispatch:
    on_bar: Callable[[Bar], Iterable[Order] | None] | None
    on_bars: BatchHook | None
    drain_pending: Callable[[], list[Order]] | None
    on_before_bars: BatchHook | None
    before_trading_day: Callable[[date, list[Bar]], None] | None
    set_market_slice: Callable[[list[Bar]], None] | None
    notify_order: Callable[[OrderRecord], None] | None


def _optional_hook(strategy: object, name: str) -> Any:
    hook = getattr(strategy, name, None)
    if hook is not None and not callable(hook):
        raise TypeError(f"Strategy hook {name} must be callable or None")
    return hook


def bind_strategy(strategy: StrategyHooks, broker: BrokerSimulator) -> StrategyDispatch:
    """Validate lifecycle and optional hooks before entering the hot loop."""
    for name in ("initialize", "finalize"):
        if not callable(getattr(strategy, name, None)):
            raise TypeError(f"Strategy lifecycle hook {name} must be callable")
    # Base Strategy's internal hooks are required for order draining and context.
    # Explicit member access lets static checking catch accidental renames.
    if isinstance(strategy, Strategy):
        strategy._bind_broker(broker)
        drain = strategy._drain_pending_orders
        market_slice = strategy._set_market_slice
    else:
        binder = _optional_hook(strategy, "_bind_broker")
        if binder is not None:
            binder(broker)
        drain = _optional_hook(strategy, "_drain_pending_orders")
        market_slice = _optional_hook(strategy, "_set_market_slice")
    notifier = _optional_hook(strategy, "notify_order")
    if getattr(notifier, "__func__", None) is Strategy.notify_order:
        notifier = None
    if isinstance(strategy, Strategy):
        user_notifier = notifier
        def notify_with_algo(record):
            strategy._observe_algo_order(record)
            if user_notifier is not None:
                user_notifier(record)
        notifier = notify_with_algo
    return StrategyDispatch(
        on_bar=_optional_hook(strategy, "on_bar"),
        on_bars=cast(BatchHook | None, _optional_hook(strategy, "on_bars")),
        drain_pending=drain,
        on_before_bars=cast(BatchHook | None, _optional_hook(strategy, "on_before_bars")),
        before_trading_day=_optional_hook(strategy, "before_trading_day"),
        set_market_slice=market_slice,
        notify_order=notifier,
    )
