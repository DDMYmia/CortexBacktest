"""Composable risk controls for event-driven backtests."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Protocol

from ..contracts import QUANTITY_TOLERANCE, Order, Position, Side
from .ledger import AccountState


@dataclass(frozen=True, slots=True)
class RiskContext:
    order: Order
    state: AccountState
    price: float | None = None

    @property
    def order_value(self) -> float | None:
        if self.price is None:
            return None
        multiplier = self.state.contract_multipliers.get(self.order.symbol.upper(), 1.0)
        return abs(self.order.quantity * self.price * multiplier)


class RiskRule(Protocol):
    def check(self, context: RiskContext) -> str | None:
        """Return a rejection reason, or None if the order passes."""


class RiskEngine:
    def __init__(self, rules: list[RiskRule] | None = None) -> None:
        self.rules = rules or []

    @property
    def has_rules(self) -> bool:
        return bool(self.rules)

    def check(self, context: RiskContext) -> str | None:
        for rule in self.rules:
            reason = rule.check(context)
            if reason is not None:
                return reason
        return None


@dataclass(frozen=True, slots=True)
class LongOnlyRule:
    """Reject sell orders that would create or increase a short position."""

    def check(self, context: RiskContext) -> str | None:
        if context.order.side != Side.SELL:
            return None
        current = context.state.positions.get(context.order.symbol)
        current_qty = current.quantity if current else 0.0
        if context.order.quantity > current_qty:
            return "long-only rule rejected short sale"
        return None


@dataclass(frozen=True, slots=True)
class MaxOrderValueRule:
    max_value: float

    def __post_init__(self) -> None:
        _require_positive(self.max_value, "max_value")

    def check(self, context: RiskContext) -> str | None:
        value = context.order_value
        if value is not None and value > self.max_value:
            return f"order value exceeds max_value {self.max_value}"
        return None


@dataclass(frozen=True, slots=True)
class MaxPositionValueRule:
    max_value: float

    def __post_init__(self) -> None:
        _require_positive(self.max_value, "max_value")

    def check(self, context: RiskContext) -> str | None:
        if context.price is None:
            return None
        projected = _projected_position(context)
        if abs(_position_value(projected, context.state)) > self.max_value:
            return f"position value exceeds max_value {self.max_value}"
        return None


@dataclass(frozen=True, slots=True)
class MaxLeverageRule:
    max_leverage: float

    def __post_init__(self) -> None:
        _require_positive(self.max_leverage, "max_leverage")

    def check(self, context: RiskContext) -> str | None:
        if context.price is None:
            return None
        equity = projected_equity(context)
        if equity <= 0:
            return "equity must be positive for leverage check"
        gross_exposure = projected_gross_exposure(context)
        leverage = gross_exposure / equity
        if leverage > self.max_leverage:
            return f"leverage exceeds max_leverage {self.max_leverage}"
        return None


@dataclass(frozen=True, slots=True)
class MaxPositionWeightRule:
    max_weight: float

    def __post_init__(self) -> None:
        _require_positive(self.max_weight, "max_weight")

    def check(self, context: RiskContext) -> str | None:
        if context.price is None:
            return None
        equity = projected_equity(context)
        if equity <= 0:
            return "equity must be positive for position weight check"
        projected = _projected_position(context)
        weight = abs(_position_value(projected, context.state)) / equity
        if weight > self.max_weight:
            return f"position weight exceeds max_weight {self.max_weight}"
        return None


@dataclass(frozen=True, slots=True)
class MaxGrossExposureRule:
    max_exposure: float

    def __post_init__(self) -> None:
        _require_positive(self.max_exposure, "max_exposure")

    def check(self, context: RiskContext) -> str | None:
        if context.price is None:
            return None
        gross_exposure = projected_gross_exposure(context)
        if gross_exposure > self.max_exposure:
            return f"gross exposure exceeds max_exposure {self.max_exposure}"
        return None


@dataclass(frozen=True, slots=True)
class MaxNetExposureRule:
    max_abs_exposure: float

    def __post_init__(self) -> None:
        _require_positive(self.max_abs_exposure, "max_abs_exposure")

    def check(self, context: RiskContext) -> str | None:
        if context.price is None:
            return None
        net_exposure = abs(projected_net_exposure(context))
        if net_exposure > self.max_abs_exposure:
            return f"net exposure exceeds max_abs_exposure {self.max_abs_exposure}"
        return None


@dataclass(frozen=True, slots=True)
class MaxGroupWeightRule:
    groups: dict[str, str]
    max_weights: dict[str, float]
    default_group: str = "ungrouped"

    def __post_init__(self) -> None:
        if not self.max_weights:
            raise ValueError("max_weights must not be empty")
        invalid = {
            group for group, weight in self.max_weights.items()
            if not isfinite(weight) or weight <= 0
        }
        if invalid:
            raise ValueError(f"group max weights must be positive and finite: {sorted(invalid)}")

    def check(self, context: RiskContext) -> str | None:
        if context.price is None:
            return None
        equity = projected_equity(context)
        if equity <= 0:
            return "equity must be positive for group weight check"
        exposures = projected_group_exposures(context, self.groups, default_group=self.default_group)
        for group, max_weight in self.max_weights.items():
            weight = exposures.get(group, 0.0) / equity
            if weight > max_weight:
                return f"group {group} weight exceeds max_weight {max_weight}"
        return None


@dataclass(frozen=True, slots=True)
class MarginRequirementRule:
    """Reject orders whose projected gross exposure exceeds margin capacity."""

    initial_margin: float = 1.0

    def __post_init__(self) -> None:
        _require_positive(self.initial_margin, "initial_margin")

    def check(self, context: RiskContext) -> str | None:
        if context.price is None:
            return None
        equity = projected_equity(context)
        if equity <= 0:
            return "equity must be positive for margin check"
        required_margin = projected_gross_exposure(context) * self.initial_margin
        if required_margin > equity:
            return f"margin requirement exceeds equity at initial_margin {self.initial_margin}"
        return None


def projected_positions(context: RiskContext) -> dict[str, Position]:
    positions = dict(context.state.positions)
    projected = _projected_position(context)
    if abs(projected.quantity) < QUANTITY_TOLERANCE:
        positions.pop(projected.symbol, None)
    else:
        positions[projected.symbol] = projected
    return positions


def projected_equity(context: RiskContext) -> float:
    prices = {context.order.symbol: context.price} if context.price is not None else None
    return context.state.equity(prices)


def projected_gross_exposure(context: RiskContext) -> float:
    return sum(abs(_position_value(position, context.state)) for position in projected_positions(context).values())


def projected_net_exposure(context: RiskContext) -> float:
    return sum(_position_value(position, context.state) for position in projected_positions(context).values())


def projected_group_exposures(
    context: RiskContext,
    groups: dict[str, str],
    *,
    default_group: str = "ungrouped",
) -> dict[str, float]:
    exposures: dict[str, float] = {}
    normalized_groups = {str(symbol).upper(): group for symbol, group in groups.items()}
    for symbol, position in projected_positions(context).items():
        group = normalized_groups.get(symbol.upper(), default_group)
        exposures[group] = exposures.get(group, 0.0) + abs(_position_value(position, context.state))
    return exposures


def _projected_position(context: RiskContext) -> Position:
    current = context.state.positions.get(context.order.symbol)
    current_qty = current.quantity if current else 0.0
    signed_qty = context.order.quantity if context.order.side == Side.BUY else -context.order.quantity
    price = context.price if context.price is not None else (current.last_price if current else None)
    if price is None:
        raise ValueError("price is required to project a new position")
    return Position(context.order.symbol, current_qty + signed_qty, price)


def _position_value(position: Position, state: AccountState) -> float:
    return position.market_value * state.contract_multipliers.get(position.symbol.upper(), 1.0)


def _require_positive(value: float, name: str) -> None:
    if not isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")
