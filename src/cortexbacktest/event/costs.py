"""Commission and slippage models for event-driven backtests."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from ..contracts import Bar, Order, Side


class CommissionModel:
    def calculate(self, order: Order, price: float, *, contract_multiplier: float = 1.0) -> float:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class PercentCommission(CommissionModel):
    bps: float = 0.0
    minimum: float = 0.0

    def calculate(self, order: Order, price: float, *, contract_multiplier: float = 1.0) -> float:
        if any(not isfinite(value) or value < 0 for value in (self.bps, self.minimum)):
            raise ValueError("commission inputs must be finite and non-negative")
        commission = abs(order.quantity * price * contract_multiplier) * (self.bps / 10_000.0)
        return max(self.minimum, commission) if commission > 0 else 0.0


@dataclass(frozen=True, slots=True)
class FixedCommission(CommissionModel):
    amount: float = 0.0

    def calculate(self, order: Order, price: float, *, contract_multiplier: float = 1.0) -> float:
        if not isfinite(self.amount) or self.amount < 0:
            raise ValueError("amount must be finite and non-negative")
        return self.amount if order.quantity > 0 else 0.0


@dataclass(frozen=True, slots=True)
class PerShareCommission(CommissionModel):
    amount_per_share: float = 0.0
    minimum: float = 0.0

    def calculate(self, order: Order, price: float, *, contract_multiplier: float = 1.0) -> float:
        if any(not isfinite(value) or value < 0 for value in (self.amount_per_share, self.minimum)):
            raise ValueError("commission inputs must be finite and non-negative")
        commission = abs(order.quantity) * self.amount_per_share
        return max(self.minimum, commission) if commission > 0 else 0.0


@dataclass(frozen=True, slots=True)
class TieredCommission(CommissionModel):
    tiers: tuple[tuple[float, float], ...]
    minimum: float = 0.0

    def calculate(self, order: Order, price: float, *, contract_multiplier: float = 1.0) -> float:
        if not isfinite(self.minimum) or self.minimum < 0:
            raise ValueError("minimum must be finite and non-negative")
        remaining = abs(order.quantity)
        if remaining <= 0:
            return 0.0
        previous_limit = 0.0
        commission = 0.0
        for limit, amount_per_share in self.tiers:
            if not limit > previous_limit or not isfinite(amount_per_share) or amount_per_share < 0:
                raise ValueError("tiers must have increasing limits and finite non-negative rates")
            tranche = min(remaining, limit - previous_limit)
            if tranche > 0:
                commission += tranche * amount_per_share
                remaining -= tranche
            previous_limit = limit
            if remaining <= 0:
                break
        if remaining > 0:
            if not self.tiers:
                raise ValueError("tiers must not be empty")
            commission += remaining * self.tiers[-1][1]
        return max(self.minimum, commission)


@dataclass(frozen=True, slots=True)
class CommissionScheme:
    """Bundle commission and margin settings without changing broker hot paths."""

    commission_model: CommissionModel
    initial_margin: float = 1.0
    maintenance_margin: float | None = None

    def __post_init__(self) -> None:
        if not isfinite(self.initial_margin) or self.initial_margin <= 0:
            raise ValueError("initial_margin must be finite and positive")
        if self.maintenance_margin is not None and (
            not isfinite(self.maintenance_margin) or self.maintenance_margin <= 0
        ):
            raise ValueError("maintenance_margin must be finite and positive")


class SlippageModel:
    def adjust_price(self, order: Order, bar: Bar, base_price: float) -> float:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class BasisPointSlippage(SlippageModel):
    bps: float = 0.0

    def adjust_price(self, order: Order, bar: Bar, base_price: float) -> float:
        if not isfinite(self.bps) or self.bps < 0:
            raise ValueError("slippage bps must be finite and non-negative")
        adjustment = base_price * (self.bps / 10_000.0)
        return base_price + adjustment if order.side == Side.BUY else base_price - adjustment


@dataclass(frozen=True, slots=True)
class VolumeShareSlippage(SlippageModel):
    """Slippage that grows with order participation in bar volume."""

    fixed_bps: float = 0.0
    impact_bps: float = 0.0
    power: float = 1.0

    def adjust_price(self, order: Order, bar: Bar, base_price: float) -> float:
        if any(not isfinite(value) or value < 0 for value in (self.fixed_bps, self.impact_bps)):
            raise ValueError("slippage bps inputs must be finite and non-negative")
        if not isfinite(self.power) or self.power <= 0:
            raise ValueError("power must be finite and positive")
        participation = 0.0
        if bar.volume is not None and bar.volume > 0:
            participation = min(abs(order.quantity) / float(bar.volume), 1.0)
        total_bps = self.fixed_bps + self.impact_bps * (participation**self.power)
        adjustment = base_price * (total_bps / 10_000.0)
        return base_price + adjustment if order.side == Side.BUY else base_price - adjustment


@dataclass(frozen=True, slots=True)
class SquareRootMarketImpact(SlippageModel):
    """Temporary market impact using the common square-root participation law.

    ``impact_bps`` is the impact at full participation.  The model is still a
    bar-level approximation: it is intended for capacity and robustness tests,
    not for replacing a historical order-book simulator.
    """

    fixed_bps: float = 0.0
    impact_bps: float = 0.0
    max_participation: float = 1.0

    def __post_init__(self) -> None:
        if any(not isfinite(value) or value < 0 for value in (self.fixed_bps, self.impact_bps)):
            raise ValueError("market impact bps inputs must be finite and non-negative")
        if not 0 < self.max_participation <= 1:
            raise ValueError("max_participation must be in (0, 1]")

    def adjust_price(self, order: Order, bar: Bar, base_price: float) -> float:
        participation = 0.0
        if bar.volume is not None and bar.volume > 0:
            participation = min(abs(order.quantity) / float(bar.volume), self.max_participation)
        total_bps = self.fixed_bps + self.impact_bps * (participation ** 0.5)
        adjustment = base_price * (total_bps / 10_000.0)
        return base_price + adjustment if order.side == Side.BUY else base_price - adjustment
