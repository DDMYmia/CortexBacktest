from datetime import UTC, datetime

from cortexbacktest import Bar
from cortexbacktest.contracts import Fill, Side
from cortexbacktest.event import Ledger


def test_ledger_applies_fill_and_marks_position():
    ledger = Ledger(initial_cash=1_000)
    fill = Fill("AAPL", datetime(2024, 1, 1, tzinfo=UTC), Side.BUY, 5, 100, commission=1)

    ledger.apply_fill(fill)
    ledger.mark(Bar("AAPL", datetime(2024, 1, 2, tzinfo=UTC), 110, 110, 110, 110))

    assert ledger.state.cash == 499
    assert ledger.state.positions["AAPL"].quantity == 5
    assert ledger.state.positions["AAPL"].last_price == 110
    assert ledger.state.equity() == 1_049


def test_ledger_sell_can_close_position():
    ledger = Ledger(initial_cash=1_000)
    ledger.apply_fill(Fill("AAPL", datetime(2024, 1, 1, tzinfo=UTC), Side.BUY, 5, 100))
    ledger.apply_fill(Fill("AAPL", datetime(2024, 1, 2, tzinfo=UTC), Side.SELL, 5, 120))

    assert ledger.state.cash == 1_100
    assert "AAPL" not in ledger.state.positions
    assert len(ledger.fills) == 2


def test_ledger_can_fill_checks_cash_and_short_rules():
    ledger = Ledger(initial_cash=100)

    assert ledger.can_fill(
        side=Side.BUY,
        symbol="AAPL",
        quantity=2,
        value=200,
        commission=0,
        allow_short=True,
    ) == "insufficient cash"
    assert ledger.can_fill(
        side=Side.SELL,
        symbol="AAPL",
        quantity=1,
        value=100,
        commission=0,
        allow_short=False,
    ) == "short selling is disabled"


def test_equity_matches_snapshot_for_lowercase_multiplier_symbol():
    # Contract multipliers are registered uppercase, so both equity paths must
    # normalize the symbol before looking one up. A lowercase fill symbol used
    # to be worth 1x in equity() and 50x in equity_snapshot_row().
    ledger = Ledger(initial_cash=1_000_000, contract_multipliers={"es": 50})
    ledger.apply_fill(Fill("es", datetime(2024, 1, 1, tzinfo=UTC), Side.BUY, 2, 5_000))
    ledger.mark(Bar("es", datetime(2024, 1, 2, tzinfo=UTC), 5_100, 5_100, 5_100, 5_100))

    snapshot = ledger.equity_snapshot(datetime(2024, 1, 2, tzinfo=UTC))

    assert ledger.state.equity() == snapshot["equity"]
    assert ledger.state.equity() == 1_010_000
