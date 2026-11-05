"""Research collateral model with stock cover and same-expiry option netting.

Unlike the unnetted ResearchOptionMargin, this model pairs verticals, combines
call/put spread charges and nets short combinations. Pairing rules are below.
It is not broker-certified or portfolio margin; it has no cross-expiry,
cross-underlying or delta-based offsets. Arbitrary overlapping structures and
mixed contract terms are not certified by the standard spread examples.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Mapping

from .contracts import OptionContract

# Pairing order: expiry, then strike, then symbol — so a book with several
# identical legs always nets the same way, whatever order the positions dict
# happens to be in.
_ORDER = lambda leg: (leg.contract.expiry, leg.contract.strike, leg.contract.symbol)  # noqa: E731


@dataclass(slots=True)
class _Leg:
    contract: OptionContract
    quantity: float
    premium: float


def _requirement_naked(contract: OptionContract, spot: float, premium: float,
                       equity_fraction: float, minimum_fraction: float) -> float:
    """Reg-T naked short: premium + max(pct of spot less out-of-the-money, floor).

    The model uses spot for the call floor and strike for the put floor.
    """
    if contract.right == "CALL":
        out_of_money = max(contract.strike - spot, 0.0)
        floor_base = spot
    else:
        out_of_money = max(spot - contract.strike, 0.0)
        floor_base = contract.strike
    return premium + max(equity_fraction * spot - out_of_money, minimum_fraction * floor_base)


@dataclass(frozen=True, slots=True)
class RegTOptionMargin:
    """Reg-T style collateral with defined-risk netting. Research model, not a broker.

    `equity_fraction`/`minimum_fraction` are the naked-short percentages (20% of
    spot less the out-of-the-money amount, floored at 10%); index options use
    lower numbers, so they are parameters rather than constants.
    `stock_fraction` is the requirement on a stock position (0.50 initial, 0.25
    maintenance under Reg-T). `leverage` names leveraged ETFs and their factor: a
    3x fund's stock and naked-short percentages are three times the plain ones,
    capped at 100% in this model.

    Netting, applied in this order per underlying:

    1. **Stock cover** — long stock covers short calls, short stock covers short
       puts, `multiplier` shares per contract. A covered call requires nothing
       beyond the stock's own margin.
    2. **Vertical spreads** — a short leg paired with a long leg of the same
       right and expiry costs the strike width when the long is the wider strike
       (a credit spread's maximum loss), and the long leg's premium when it is
       the deeper one (a debit spread, which cannot lose more than it paid, but
       whose long leg still has no loan value). Pairing is greedy: protective
       strikes first, then nearest strike.
    3. **Spread groups** — only the greater call/put charge at an expiry is used.
       This approximates separated iron-condor wings; strike ordering is not
       checked, so it does not establish a bound for arbitrary overlapping legs.
    4. **Short combinations** — an unpaired short call and short put at the same
       expiry are charged as the greater naked requirement plus the other side's
       premium, the standard straddle/strangle rule.
    5. Anything left is naked, and long premium is charged in full.

    The requirement is collateral to post, not the position's economic loss: a
    credit spread's width is charged while its credit sits in the account, so the
    net effect on free collateral is width minus credit.
    """

    equity_fraction: float = 0.20
    minimum_fraction: float = 0.10
    stock_fraction: float = 0.50
    leverage: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not all(isfinite(v) and 0 <= v <= 1 for v in
                   (self.equity_fraction, self.minimum_fraction, self.stock_fraction)):
            raise ValueError("margin fractions must be in [0,1]")
        if not all(isfinite(v) and v >= 1 for v in self.leverage.values()):
            raise ValueError("leverage factors must be at least 1")

    def _scaled(self, fraction: float, underlying: str) -> float:
        return min(fraction * self.leverage.get(underlying, 1.0), 1.0)

    def requirement(self, positions: Mapping[str, float],
                    contracts: Mapping[str, OptionContract],
                    marks: Mapping[str, float]) -> float:
        shorts: dict[str, list[_Leg]] = {}
        longs: dict[str, list[_Leg]] = {}
        total = 0.0
        for symbol, quantity in positions.items():
            if not quantity:
                continue
            if symbol not in contracts:
                total += abs(quantity) * marks[symbol] * self._scaled(self.stock_fraction, symbol)
                continue
            contract = contracts[symbol]
            leg = _Leg(contract, abs(quantity), marks[symbol])
            (shorts if quantity < 0 else longs).setdefault(contract.underlying, []).append(leg)

        for underlying in sorted(set(shorts) | set(longs)):
            total += self._underlying_requirement(
                underlying, shorts.get(underlying, []), longs.get(underlying, []),
                positions, marks)
        return total

    def _underlying_requirement(self, underlying: str, shorts: list[_Leg], longs: list[_Leg],
                                positions: Mapping[str, float],
                                marks: Mapping[str, float]) -> float:
        spot = marks[underlying]
        shares = float(positions.get(underlying, 0.0))
        equity, minimum = self._scaled(self.equity_fraction, underlying), self._scaled(self.minimum_fraction, underlying)
        shorts = sorted(shorts, key=_ORDER)
        longs = sorted(longs, key=_ORDER)
        total = 0.0

        # 1. Stock cover.
        for leg in shorts:
            if not leg.quantity:
                continue
            covered_shares = shares * leg.contract.right.sign
            if covered_shares <= 0:
                continue
            covered = min(leg.quantity, covered_shares / leg.contract.multiplier)
            leg.quantity -= covered
            shares -= covered * leg.contract.multiplier * leg.contract.right.sign

        # 2. Vertical spreads, then 3. condor netting, per expiry.
        spread_charges: dict[tuple[object, str], float] = {}
        for short in shorts:
            while short.quantity > 0:
                protector = self._nearest_protective(short, longs)
                if protector is None:
                    break
                matched = min(short.quantity, protector.quantity)
                short.quantity -= matched
                protector.quantity -= matched
                width = abs(protector.contract.strike - short.contract.strike)
                protective = (protector.contract.strike > short.contract.strike
                              if short.contract.right == "CALL"
                              else protector.contract.strike < short.contract.strike)
                key = (short.contract.expiry, short.contract.right)
                if protective:
                    # The width is the pair's entire maximum loss, so it replaces
                    # both the naked charge and the long leg's premium.
                    charge = matched * short.contract.multiplier * width
                else:
                    # The long leg is deeper in the money than the short: a debit
                    # spread, whose loss cannot exceed what was already paid for
                    # the long. It is still charged its own premium, because a
                    # long option has no loan value and must not become free
                    # collateral for something else.
                    charge = matched * short.contract.multiplier * protector.premium
                spread_charges[key] = spread_charges.get(key, 0.0) + charge
        for expiry in {key[0] for key in spread_charges}:
            calls = spread_charges.get((expiry, "CALL"), 0.0)
            puts = spread_charges.get((expiry, "PUT"), 0.0)
            total += max(calls, puts) if calls and puts else calls + puts

        # 4. Short combinations at one expiry.
        remaining = [leg for leg in shorts if leg.quantity > 0]
        by_expiry: dict[object, dict[str, list[_Leg]]] = {}
        for leg in remaining:
            by_expiry.setdefault(leg.contract.expiry, {}).setdefault(leg.contract.right, []).append(leg)
        for expiry in sorted(by_expiry, key=str):
            sides = by_expiry[expiry]
            call_legs, put_legs = sides.get("CALL", []), sides.get("PUT", [])
            while call_legs and put_legs:
                call, put = call_legs[0], put_legs[0]
                matched = min(call.quantity, put.quantity)
                call.quantity -= matched
                put.quantity -= matched
                call_requirement = _requirement_naked(call.contract, spot, call.premium, equity, minimum)
                put_requirement = _requirement_naked(put.contract, spot, put.premium, equity, minimum)
                # The model charges the greater naked side plus the other premium.
                total += matched * call.contract.multiplier * (
                    max(call_requirement, put_requirement)
                    + (put.premium if call_requirement >= put_requirement else call.premium))
                if not call.quantity:
                    call_legs.pop(0)
                if not put.quantity:
                    put_legs.pop(0)

        # 5. Whatever is still short is naked, and unpaired long premium is
        # charged in full — a long option is paid for, so it collateralises
        # exactly its own market value and nothing more.
        total += sum(leg.quantity * leg.contract.multiplier * leg.premium
                     for leg in longs if leg.quantity > 0)
        for leg in shorts:
            if leg.quantity > 0:
                total += leg.quantity * leg.contract.multiplier * _requirement_naked(
                    leg.contract, spot, leg.premium, equity, minimum)
        return total

    @staticmethod
    def _nearest_protective(short: _Leg, longs: list[_Leg]) -> _Leg | None:
        """The available long leg that caps this short's loss most tightly.

        Same right and same expiry only: a calendar's long leg expires after the
        short and does not cap the short's assignment risk at its own expiry, and
        Reg-T does not let one expiry protect another here.
        """
        candidates = [leg for leg in longs
                      if leg.quantity > 0
                      and leg.contract.right == short.contract.right
                      and leg.contract.expiry == short.contract.expiry
                      and leg.contract.multiplier == short.contract.multiplier]
        def distance(leg: _Leg) -> tuple[int, float, float, str]:
            width = abs(leg.contract.strike - short.contract.strike)
            protective = (leg.contract.strike > short.contract.strike
                          if short.contract.right == "CALL"
                          else leg.contract.strike < short.contract.strike)
            # Protective legs first, nearest strike first, then deterministic.
            return (0 if protective else 1, width, leg.contract.strike, leg.contract.symbol)
        return min(candidates, key=distance, default=None)
