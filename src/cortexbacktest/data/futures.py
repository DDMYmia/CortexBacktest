"""Continuous futures construction and back-adjustment utilities."""

from __future__ import annotations

from copy import replace
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Literal

from ..contracts import Bar

RollAdjustment = Literal["none", "difference", "ratio"]


@dataclass(frozen=True, slots=True)
class ContractRoll:
    """A transition where ``to_contract`` becomes active at ``timestamp``.

    ``price_difference`` is added to older prices for difference adjustment.
    ``price_ratio`` multiplies older prices for ratio adjustment. When omitted,
    either value is inferred from the contracts' nearest closes around the roll.
    """

    timestamp: datetime
    from_contract: str
    to_contract: str
    price_difference: float | None = None
    price_ratio: float | None = None


def build_continuous_futures_bars(
    bars: Iterable[Bar],
    rolls: Iterable[ContractRoll],
    *,
    continuous_symbol: str,
    adjustment: RollAdjustment = "difference",
) -> list[Bar]:
    """Select active contracts and build a back-adjusted continuous series.

    The first roll's ``from_contract`` is active before its timestamp and each
    ``to_contract`` is active from its roll timestamp until the next roll.
    Historical OHLC, adjusted close, and VWAP prices are adjusted; volume and
    open interest remain unchanged.
    """
    if adjustment not in {"none", "difference", "ratio"}:
        raise ValueError("adjustment must be one of: none, difference, ratio")
    output_symbol = continuous_symbol.strip().upper()
    if not output_symbol:
        raise ValueError("continuous_symbol must not be empty")

    source = sorted(bars, key=lambda bar: (bar.timestamp, bar.symbol.upper()))
    schedule = sorted(rolls, key=lambda roll: roll.timestamp)
    if not source:
        return []
    if not schedule:
        symbols = {bar.symbol.upper() for bar in source}
        if len(symbols) != 1:
            raise ValueError("rolls are required when bars contain multiple contracts")
        return [replace(bar, symbol=output_symbol) for bar in source]

    _validate_roll_schedule(schedule)
    bars_by_symbol: dict[str, list[Bar]] = {}
    for bar in source:
        bars_by_symbol.setdefault(bar.symbol.upper(), []).append(bar)
    _validate_contract_data(schedule, bars_by_symbol)

    selected = _select_active_bars(source, schedule)
    adjustments = _resolve_adjustments(schedule, bars_by_symbol, adjustment)
    return [
        _adjust_bar(bar, output_symbol, schedule, adjustments, adjustment)
        for bar in selected
    ]


def _validate_roll_schedule(schedule: list[ContractRoll]) -> None:
    for index, roll in enumerate(schedule):
        if not roll.from_contract.strip() or not roll.to_contract.strip():
            raise ValueError("roll contract symbols must not be empty")
        if roll.from_contract.upper() == roll.to_contract.upper():
            raise ValueError("roll must transition between different contracts")
        if index and schedule[index - 1].timestamp == roll.timestamp:
            raise ValueError("roll timestamps must be unique")
        if index and schedule[index - 1].to_contract.upper() != roll.from_contract.upper():
            raise ValueError("roll schedule must form a contiguous contract chain")


def _validate_contract_data(
    schedule: list[ContractRoll], bars_by_symbol: dict[str, list[Bar]]
) -> None:
    required = {schedule[0].from_contract.upper()}
    required.update(roll.to_contract.upper() for roll in schedule)
    missing = sorted(required.difference(bars_by_symbol))
    if missing:
        raise ValueError(f"missing bars for roll contracts: {', '.join(missing)}")


def _select_active_bars(source: list[Bar], schedule: list[ContractRoll]) -> list[Bar]:
    active = schedule[0].from_contract.upper()
    roll_index = 0
    selected: list[Bar] = []
    seen_timestamps: set[datetime] = set()
    for bar in source:
        while roll_index < len(schedule) and bar.timestamp >= schedule[roll_index].timestamp:
            active = schedule[roll_index].to_contract.upper()
            roll_index += 1
        if bar.symbol.upper() != active:
            continue
        if bar.timestamp in seen_timestamps:
            raise ValueError(f"duplicate active-contract bar at {bar.timestamp.isoformat()}")
        selected.append(bar)
        seen_timestamps.add(bar.timestamp)
    return selected


def _resolve_adjustments(
    schedule: list[ContractRoll],
    bars_by_symbol: dict[str, list[Bar]],
    adjustment: RollAdjustment,
) -> list[float]:
    if adjustment == "none":
        return [0.0] * len(schedule)
    values: list[float] = []
    for roll in schedule:
        value = roll.price_difference if adjustment == "difference" else roll.price_ratio
        if value is None:
            old_close = _nearest_close(
                bars_by_symbol[roll.from_contract.upper()], roll.timestamp, before=True
            )
            new_close = _nearest_close(
                bars_by_symbol[roll.to_contract.upper()], roll.timestamp, before=False
            )
            if adjustment == "ratio" and old_close == 0:
                raise ValueError("cannot infer ratio adjustment from a zero close")
            value = new_close - old_close if adjustment == "difference" else new_close / old_close
        values.append(value)
    return values


def _nearest_close(bars: list[Bar], timestamp: datetime, *, before: bool) -> float:
    if before:
        candidates = (bar for bar in bars if bar.timestamp < timestamp)
    else:
        candidates = (bar for bar in bars if bar.timestamp >= timestamp)
    bar = max(candidates, key=lambda item: item.timestamp, default=None) if before else min(
        candidates, key=lambda item: item.timestamp, default=None
    )
    if bar is None:
        side = "before" if before else "at or after"
        raise ValueError(f"cannot infer roll adjustment: no contract close {side} roll")
    return bar.close


def _adjust_bar(
    bar: Bar,
    symbol: str,
    schedule: list[ContractRoll],
    adjustments: list[float],
    method: RollAdjustment,
) -> Bar:
    applicable = [
        value
        for roll, value in zip(schedule, adjustments, strict=True)
        if bar.timestamp < roll.timestamp
    ]
    shift = 0.0
    factor = 1.0
    if method == "difference":
        shift = sum(applicable)
    elif method == "ratio":
        for value in applicable:
            factor *= value

    def transform(value: float) -> float:
        if method == "difference":
            return value + shift
        if method == "ratio":
            return value * factor
        return value

    return replace(
        bar,
        symbol=symbol,
        open=transform(bar.open),
        high=transform(bar.high),
        low=transform(bar.low),
        close=transform(bar.close),
        adj_close=transform(bar.adj_close) if bar.adj_close is not None else None,
        vwap=transform(bar.vwap) if bar.vwap is not None else None,
    )
