"""Compare fixed bps slippage with volume-share impact slippage."""

from __future__ import annotations

from datetime import UTC, datetime

from cortexbacktest import Bar, BasisPointSlippage, Order, Side, VolumeShareSlippage
from cortexbacktest.broker import BrokerSimulator


def main() -> None:
    timestamp = datetime(2024, 1, 1, tzinfo=UTC)
    bar = Bar("AAPL", timestamp, open=100, high=100, low=100, close=100, volume=1_000)
    order = Order("AAPL", timestamp, Side.BUY, 300)

    fixed = BrokerSimulator(slippage_model=BasisPointSlippage(10)).execute(order, bar)
    impact = BrokerSimulator(slippage_model=VolumeShareSlippage(fixed_bps=0, impact_bps=100)).execute(order, bar)

    print("CortexBacktest - market impact slippage demo")
    print(f"fixed_bps_price: {fixed.price:.4f}")
    print(f"volume_share_price: {impact.price:.4f}")
    print(f"volume_share_slippage: {impact.slippage:.4f}")


if __name__ == "__main__":
    main()
