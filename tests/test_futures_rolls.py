from datetime import UTC, datetime, timedelta

import pytest

from cortexbacktest import Bar, ContractRoll, build_continuous_futures_bars


def _bar(symbol: str, day: int, close: float, *, volume: float, open_interest: float) -> Bar:
    timestamp = datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=day - 1)
    return Bar(
        symbol,
        timestamp,
        close - 1,
        close + 1,
        close - 2,
        close,
        volume=volume,
        adj_close=close - 0.25,
        vwap=close - 0.5,
        open_interest=open_interest,
    )


def test_build_difference_adjusted_continuous_futures_bars():
    bars = [
        _bar("ESH4", 1, 99, volume=10, open_interest=1_000),
        _bar("ESH4", 2, 100, volume=11, open_interest=900),
        _bar("ESM4", 2, 110, volume=20, open_interest=800),
        _bar("ESM4", 3, 111, volume=21, open_interest=1_100),
        _bar("ESM4", 4, 112, volume=22, open_interest=1_200),
    ]
    rolls = [ContractRoll(datetime(2024, 1, 3, tzinfo=UTC), "ESH4", "ESM4")]

    continuous = build_continuous_futures_bars(
        reversed(bars), rolls, continuous_symbol="ES", adjustment="difference"
    )

    assert [bar.symbol for bar in continuous] == ["ES"] * 4
    assert [bar.close for bar in continuous] == [110, 111, 111, 112]
    assert continuous[1].open == 110
    assert continuous[1].adj_close == pytest.approx(110.75)
    assert continuous[1].vwap == pytest.approx(110.5)
    assert continuous[1].volume == 11
    assert continuous[1].open_interest == 900


def test_build_ratio_adjusted_continuous_futures_bars_with_explicit_ratio():
    bars = [
        _bar("CLF4", 1, 50, volume=10, open_interest=100),
        _bar("CLG4", 2, 60, volume=20, open_interest=200),
    ]
    roll = ContractRoll(
        datetime(2024, 1, 2, tzinfo=UTC),
        "CLF4",
        "CLG4",
        price_ratio=1.2,
    )

    continuous = build_continuous_futures_bars(
        bars, [roll], continuous_symbol="cl", adjustment="ratio"
    )

    assert [bar.close for bar in continuous] == [60, 60]
    assert [bar.symbol for bar in continuous] == ["CL", "CL"]


def test_continuous_futures_accumulates_multiple_roll_adjustments():
    bars = [
        _bar("F1", 1, 100, volume=1, open_interest=1),
        _bar("F2", 2, 110, volume=1, open_interest=1),
        _bar("F3", 3, 130, volume=1, open_interest=1),
    ]
    rolls = [
        ContractRoll(datetime(2024, 1, 2, tzinfo=UTC), "F1", "F2", price_difference=10),
        ContractRoll(datetime(2024, 1, 3, tzinfo=UTC), "F2", "F3", price_difference=20),
    ]

    continuous = build_continuous_futures_bars(bars, rolls, continuous_symbol="F")

    assert [bar.close for bar in continuous] == [130, 130, 130]


def test_continuous_futures_rejects_non_contiguous_roll_chain():
    bars = [
        _bar("F1", 1, 100, volume=1, open_interest=1),
        _bar("F2", 2, 110, volume=1, open_interest=1),
        _bar("F3", 3, 120, volume=1, open_interest=1),
    ]
    rolls = [
        ContractRoll(datetime(2024, 1, 2, tzinfo=UTC), "F1", "F2"),
        ContractRoll(datetime(2024, 1, 3, tzinfo=UTC), "OTHER", "F3"),
    ]

    with pytest.raises(ValueError, match="contiguous contract chain"):
        build_continuous_futures_bars(bars, rolls, continuous_symbol="F")
