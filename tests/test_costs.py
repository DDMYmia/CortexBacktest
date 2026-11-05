from datetime import UTC, datetime

import pytest

from cortexbacktest import Bar, Order, Side, SquareRootMarketImpact, VolumeShareSlippage
from cortexbacktest.broker import BrokerSimulator


def test_volume_share_slippage_increases_buy_price_by_participation():
    model = VolumeShareSlippage(fixed_bps=5, impact_bps=20, power=1)
    bar = Bar("AAPL", datetime(2024, 1, 1, tzinfo=UTC), 100, 100, 100, 100, volume=1_000)
    order = Order("AAPL", bar.timestamp, Side.BUY, 100)

    price = model.adjust_price(order, bar, 100)

    assert price == 100.07


def test_volume_share_slippage_decreases_sell_price_by_participation():
    model = VolumeShareSlippage(fixed_bps=0, impact_bps=100, power=2)
    bar = Bar("AAPL", datetime(2024, 1, 1, tzinfo=UTC), 100, 100, 100, 100, volume=1_000)
    order = Order("AAPL", bar.timestamp, Side.SELL, 500)

    price = model.adjust_price(order, bar, 100)

    assert price == 99.75


def test_volume_share_slippage_uses_fixed_bps_without_volume():
    model = VolumeShareSlippage(fixed_bps=10, impact_bps=100)
    bar = Bar("AAPL", datetime(2024, 1, 1, tzinfo=UTC), 100, 100, 100, 100)
    order = Order("AAPL", bar.timestamp, Side.BUY, 500)

    price = model.adjust_price(order, bar, 100)

    assert price == 100.1


def test_broker_can_use_volume_share_slippage_model():
    bar = Bar("AAPL", datetime(2024, 1, 1, tzinfo=UTC), 100, 100, 100, 100, volume=1_000)
    broker = BrokerSimulator(slippage_model=VolumeShareSlippage(fixed_bps=0, impact_bps=100))

    fill = broker.execute(Order("AAPL", bar.timestamp, Side.BUY, 100), bar)

    assert fill.price == 100.1
    assert round(fill.slippage, 4) == 10.0


def test_square_root_market_impact_scales_with_participation():
    model = SquareRootMarketImpact(impact_bps=100)
    bar = Bar("AAPL", datetime(2024, 1, 1, tzinfo=UTC), 100, 100, 100, 100, volume=10_000)
    order = Order("AAPL", bar.timestamp, Side.BUY, 2_500)

    assert model.adjust_price(order, bar, 100) == pytest.approx(100.5)
