from cortexbacktest import FillPrice
from datetime import UTC, datetime

import polars as pl
import pytest

from cortexbacktest import BacktestEngine, Bar, BrokerSimulator, Order, Side, Strategy
from cortexbacktest.contracts import Fill, Position
from cortexbacktest.reporting import (
    pnl_reconciliation_frame,
    pnl_reconciliation_summary,
    trade_stats,
    trades_frame,
)


def test_trades_frame_matches_long_round_trip():
    fills = [
        Fill("AAPL", datetime(2024, 1, 1, tzinfo=UTC), Side.BUY, 10, 100, commission=1),
        Fill("AAPL", datetime(2024, 1, 2, tzinfo=UTC), Side.SELL, 10, 110, commission=1),
    ]

    trades = trades_frame(fills)

    assert trades.height == 1
    row = trades.row(0, named=True)
    assert row["direction"] == "long"
    assert row["gross_pnl"] == 100
    assert row["costs"] == 2
    assert row["pnl"] == 98
    assert trade_stats(trades)["win_rate"] == 1.0


def test_trades_pl_matches_long_round_trip():
    fills = [
        Fill("AAPL", datetime(2024, 1, 1, tzinfo=UTC), Side.BUY, 10, 100, commission=1),
        Fill("AAPL", datetime(2024, 1, 2, tzinfo=UTC), Side.SELL, 10, 110, commission=1),
    ]

    trades = trades_frame(fills)

    assert isinstance(trades, pl.DataFrame)
    row = trades.row(0, named=True)
    assert row["direction"] == "long"
    assert row["gross_pnl"] == 100
    assert row["costs"] == 2
    assert row["pnl"] == 98
    assert trade_stats(trades) == trade_stats(trades.to_pandas())


def test_trades_frame_matches_short_round_trip():
    fills = [
        Fill("AAPL", datetime(2024, 1, 1, tzinfo=UTC), Side.SELL, 10, 100),
        Fill("AAPL", datetime(2024, 1, 2, tzinfo=UTC), Side.BUY, 10, 90),
    ]

    trades = trades_frame(fills)

    row = trades.row(0, named=True)
    assert row["direction"] == "short"
    assert row["pnl"] == 100


def test_trades_frame_handles_partial_fifo_closes():
    fills = [
        Fill("AAPL", datetime(2024, 1, 1, tzinfo=UTC), Side.BUY, 10, 100),
        Fill("AAPL", datetime(2024, 1, 2, tzinfo=UTC), Side.SELL, 4, 110),
        Fill("AAPL", datetime(2024, 1, 3, tzinfo=UTC), Side.SELL, 6, 90),
    ]

    trades = trades_frame(fills)

    assert trades.height == 2
    assert trades["quantity"].to_list() == [4, 6]
    assert trades["pnl"].to_list() == [40, -60]
    assert trade_stats(trades)["profit_factor"] == 40 / 60


def test_futures_fifo_scales_pnl_and_basis_but_not_cash_commissions():
    fills = [
        Fill("ES", datetime(2024, 1, 1, tzinfo=UTC), Side.BUY, 3, 100, commission=3),
        Fill("ES", datetime(2024, 1, 2, tzinfo=UTC), Side.SELL, 1, 110, commission=1),
        Fill("ES", datetime(2024, 1, 3, tzinfo=UTC), Side.SELL, 4, 90, commission=4),
        Fill("ES", datetime(2024, 1, 4, tzinfo=UTC), Side.BUY, 2, 100, commission=2),
    ]
    trades = trades_frame(fills, contract_multipliers={"es": 50})
    assert trades["direction"].to_list() == ["long", "long", "short"]
    assert trades["quantity"].to_list() == [1, 2, 2]
    assert trades["gross_pnl"].to_list() == [500, -1000, -1000]
    assert trades["costs"].to_list() == [2, 4, 4]
    assert trades["pnl"].to_list() == [498, -1004, -1004]
    assert trades["return"].to_list() == pytest.approx([498 / 5000, -1004 / 10000, -1004 / 9000])


@pytest.mark.parametrize("multiplier", [1, 5, 50])
def test_result_report_reconciles_partial_close_and_reversal(multiplier):
    class ReversePosition(Strategy):
        def on_bar(self, bar):
            side, quantity = [(Side.BUY, 3), (Side.SELL, 1), (Side.SELL, 4), (Side.BUY, 2)][bar.timestamp.day - 1]
            return [Order(bar.symbol, bar.timestamp, side, quantity)]

    bars = [
        Bar("ES", datetime(2024, 1, day, tzinfo=UTC), price, price, price, price)
        for day, price in enumerate([100, 110, 90, 100], start=1)
    ]
    result = BacktestEngine(trade_on_close=True, broker=BrokerSimulator(fill_price=FillPrice.CLOSE,
        initial_cash=100_000, commission_bps=10, contract_multipliers={"ES": multiplier},
    )).run(bars, ReversePosition())
    # Independent price-difference oracle: +10 on one long, -10 on two
    # remaining longs, then -10 on two shorts. All positions finish flat.
    expected_gross = -30 * multiplier
    cash_fees = sum(fill.commission for fill in result.fills)
    assert cash_fees > 0
    expected_net = expected_gross - cash_fees
    trades = result.trades.frame()
    assert trades["gross_pnl"].to_list() == [10 * multiplier, -20 * multiplier, -20 * multiplier]
    assert trades["costs"].sum() == pytest.approx(cash_fees)
    assert result.trades.summary()["total_pnl"] == pytest.approx(expected_net)
    report = result.report()
    assert report["trades"]["total_pnl"] == pytest.approx(expected_net)
    assert sum(row["pnl"] for row in report["records"]["trades"]) == pytest.approx(expected_net)
    assert report["pnl_reconciliation"]["total_pnl"] == pytest.approx(expected_net)
    assert report["summary"]["final_equity"] == pytest.approx(100_000 + expected_net)
    assert report["pnl_reconciliation"]["reconciliation_error"] == pytest.approx(0, abs=1e-9)


def test_pnl_reconciliation_matches_closed_round_trip_equity():
    fills = [
        Fill("AAPL", datetime(2024, 1, 1, tzinfo=UTC), Side.BUY, 10, 100, commission=1),
        Fill("AAPL", datetime(2024, 1, 2, tzinfo=UTC), Side.SELL, 10, 110, commission=1),
    ]

    summary = pnl_reconciliation_summary(fills, initial_equity=100_000, final_equity=100_098)

    assert summary["realized_gross_pnl"] == 100
    assert summary["realized_costs"] == 2
    assert summary["open_unrealized_pnl"] == 0
    assert summary["total_pnl"] == 98
    assert summary["reconciliation_error"] == 0


def test_pnl_reconciliation_marks_open_lots_with_contract_multiplier():
    fills = [
        Fill("ES", datetime(2024, 1, 1, tzinfo=UTC), Side.BUY, 2, 5000, commission=4),
    ]

    frame = pnl_reconciliation_frame(
        fills,
        positions=[Position("ES", 2, 5010)],
        initial_equity=100_000,
        final_equity=100_996,
        contract_multipliers={"ES": 50},
    )
    row = frame.row(0, named=True)

    assert row["open_unrealized_pnl"] == 1000
    assert row["commission"] == 4
    assert row["total_pnl"] == 996
    assert row["reconciliation_error"] == 0


def test_trade_stats_pl_handles_empty_and_missing_pnl():
    assert trade_stats(pl.DataFrame()) == {
        "trade_count": 0,
        "win_count": 0,
        "loss_count": 0,
        "win_rate": 0.0,
        "total_pnl": 0.0,
        "average_pnl": 0.0,
        "profit_factor": 0.0,
    }
    assert trade_stats(pl.DataFrame({"symbol": ["AAPL"]}))["trade_count"] == 0


class RoundTripStrategy(Strategy):
    def on_bar(self, bar: Bar):
        if bar.timestamp.day == 1:
            return [Order(bar.symbol, bar.timestamp, Side.BUY, 10)]
        if bar.timestamp.day == 2:
            return [Order(bar.symbol, bar.timestamp, Side.SELL, 10)]
        return []


def test_backtest_result_exposes_round_trip_trades():
    bars = [
        Bar("AAPL", datetime(2024, 1, 1, tzinfo=UTC), 100, 100, 100, 100),
        Bar("AAPL", datetime(2024, 1, 2, tzinfo=UTC), 110, 110, 110, 110),
    ]

    result = BacktestEngine(trade_on_close=True).run(bars, RoundTripStrategy())

    assert result.trades.frame().height == 1
    assert result.trades.frame().height == 1
    assert result.trades.summary()["total_pnl"] == 100
    assert result.trades.summary() == trade_stats(result.trades.frame())
    assert result.trades.pnl_summary()["reconciliation_error"] == 0
    assert result.trades.pnl_reconciliation().height == 1


def test_backtest_result_reconciles_open_position_with_initial_equity():
    bars = [
        Bar("ES", datetime(2024, 1, 1, tzinfo=UTC), 5000, 5000, 5000, 5000),
        Bar("ES", datetime(2024, 1, 2, tzinfo=UTC), 5010, 5010, 5010, 5010),
    ]

    class BuyFuture(Strategy):
        def on_bar(self, bar: Bar):
            if bar.timestamp.day == 1:
                return [Order("ES", bar.timestamp, Side.BUY, 1)]
            return []

    result = BacktestEngine(trade_on_close=True, 
        broker=BrokerSimulator(fill_price=FillPrice.CLOSE, initial_cash=300_000, contract_multipliers={"ES": 50})
    ).run(bars, BuyFuture())

    summary = result.trades.pnl_summary()
    assert result.initial_equity == 300_000
    assert summary["open_unrealized_pnl"] == 500
    assert summary["reconciliation_error"] == 0


class AlternatingStrategy(Strategy):
    """Trade in and out so realized, open, commission, and slippage all appear."""

    def on_bar(self, bar: Bar):
        if bar.timestamp.day % 3 == 1:
            return [Order(bar.symbol, bar.timestamp, Side.BUY, 10)]
        if bar.timestamp.day % 3 == 2:
            return [Order(bar.symbol, bar.timestamp, Side.SELL, 6)]
        return []


def test_pnl_reconciliation_is_exact_with_commission_and_slippage():
    # Fill prices embed slippage; reconciliation must charge commission once and
    # slippage never, or the identity to final equity breaks.
    bars = [
        Bar("AAPL", datetime(2024, 1, day, tzinfo=UTC), 100 + day, 102 + day, 99 + day, 101 + day)
        for day in range(1, 11)
    ]
    broker = BrokerSimulator(fill_price=FillPrice.CLOSE, initial_cash=100_000, commission_bps=10, slippage_bps=7)
    result = BacktestEngine(trade_on_close=True, broker=broker).run(bars, AlternatingStrategy())

    final_equity = result.performance.equity_curve()["equity"].to_list()[-1]
    summary = pnl_reconciliation_summary(
        result.fills,
        result.positions,
        initial_equity=result.initial_equity,
        final_equity=final_equity,
    )
    assert summary["slippage"] > 0
    assert summary["commission"] > 0
    assert abs(summary["reconciliation_error"]) < 1e-9
