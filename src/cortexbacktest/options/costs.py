"""Declared execution-cost assumptions, charged per filled order.

Defaults use the full quoted half-spread, 0.65 per option contract and no stock
commission. Slippage and impact are opt-in; hedging turnover comes from the
strategy's actual fills, not a separate cost parameter.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, sqrt
from typing import Protocol

from .contracts import OptionQuote


class OptionCostModel(Protocol):
    def execution(self, symbol: str, quantity: float, quote: OptionQuote, *,
                  multiplier: float, is_option: bool) -> tuple[float, float]:
        """Return the fill price and the commission for `quantity` of `symbol`.

        `quantity` is signed: positive buys and lifts the offer, negative sells
        and hits the bid. The price returned must be nonnegative and is what the
        cashbook pays; the engine attributes its distance from the quote's mid
        as execution cost.
        """
        ...


@dataclass(frozen=True, slots=True)
class ResearchOptionCosts:
    """Spread, commission, slippage and impact. A declared model, not an estimate.

    **Spread.** `spread_fraction` is how much of the quoted half-spread the fill
    pays: 0 executes at mid, 1 at the touch. Sweeping it is the point of
    `run_option_scenarios`.

    **Commission.** `option_commission` per contract, plus `option_commission_bps`
    on premium notional for the venues that charge that way, and
    `option_minimum_ticket` per order. Stock pays `stock_commission_bps` on
    notional.

    **Slippage.** `option_slippage_ticks` times `tick_size` per contract, and
    `stock_slippage_bps` on the underlying — the part of the cost that is not in
    the quote because the quote was stale, the book was thin at the touch, or the
    order crossed more than one price level. It is a declared assumption; an
    end-of-day chain cannot measure it.

    **Impact.** `impact_coefficient` times the half-spread times
    `sqrt(size / impact_reference)`, the usual square-root shape. With no
    displayed size in an end-of-day chain there is nothing to estimate it from,
    so `impact_reference` is the order size at which impact equals
    `impact_coefficient` half-spreads and both are the researcher's declaration.
    Left at zero it is off, which is honest for a one-contract study and wrong
    for a sized one.

    A fill price is floored at zero: slippage cannot make a buyer receive money.
    """

    spread_fraction: float = 1.0
    option_commission: float = 0.65
    option_commission_bps: float = 0.0
    option_minimum_ticket: float = 0.0
    option_slippage_ticks: float = 0.0
    tick_size: float = 0.01
    stock_commission_bps: float = 0.0
    stock_slippage_bps: float = 0.0
    impact_coefficient: float = 0.0
    impact_reference: float = 100.0

    def __post_init__(self) -> None:
        if not isfinite(self.spread_fraction) or not 0 <= self.spread_fraction <= 1:
            raise ValueError("spread_fraction must be in [0,1]")
        nonnegative = (self.option_commission, self.option_commission_bps,
                       self.option_minimum_ticket, self.option_slippage_ticks, self.tick_size,
                       self.stock_commission_bps, self.stock_slippage_bps,
                       self.impact_coefficient)
        if not all(isfinite(value) and value >= 0 for value in nonnegative):
            raise ValueError("cost components must be finite and nonnegative")
        if not isfinite(self.impact_reference) or self.impact_reference <= 0:
            raise ValueError("impact_reference must be positive and finite")

    def execution(self, symbol: str, quantity: float, quote: OptionQuote, *,
                  multiplier: float, is_option: bool) -> tuple[float, float]:
        if not quantity:
            raise ValueError("a fill needs a nonzero quantity")
        direction = 1.0 if quantity > 0 else -1.0
        size, mid = abs(quantity), quote.mid
        half_spread = (quote.ask - quote.bid) / 2
        offset = self.spread_fraction * half_spread
        if is_option:
            offset += self.option_slippage_ticks * self.tick_size
        else:
            offset += self.stock_slippage_bps * mid / 10_000
        if self.impact_coefficient:
            offset += self.impact_coefficient * half_spread * sqrt(size / self.impact_reference)
        price = max(0.0, mid + direction * offset)
        if is_option:
            commission = (size * self.option_commission
                          + size * multiplier * price * self.option_commission_bps / 10_000)
            commission = max(commission, self.option_minimum_ticket)
        else:
            commission = size * price * self.stock_commission_bps / 10_000
        return price, commission
