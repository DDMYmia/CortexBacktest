"""Event types for the event-driven backtester."""

from __future__ import annotations

from dataclasses import dataclass

from ..contracts import Bar, Fill, Order


@dataclass(frozen=True, slots=True)
class MarketEvent:
    bar: Bar


@dataclass(frozen=True, slots=True)
class OrderEvent:
    order: Order


@dataclass(frozen=True, slots=True)
class FillEvent:
    fill: Fill
