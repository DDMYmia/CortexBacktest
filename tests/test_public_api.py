from datetime import UTC, datetime

from cortexbacktest import (
    API_VERSION,
    BacktestConfig,
    Bar,
    BrokerConfig,
    FillPrice,
    Order,
    Side,
    Strategy,
    StrategySpec,
    run_backtest,
)


class BuyFirstBar(Strategy):
    def __init__(self) -> None:
        self.done = False

    def on_bar(self, bar):
        if self.done:
            return []
        self.done = True
        return [Order(bar.symbol, bar.timestamp, Side.BUY, 1)]


class ParameterizedBuy(Strategy):
    def __init__(self, quantity: float) -> None:
        super().__init__()
        self.quantity = quantity
        self.done = False

    def on_bar(self, bar):
        if self.done:
            return []
        self.done = True
        return [Order(bar.symbol, bar.timestamp, Side.BUY, self.quantity)]


def test_stable_api_defaults_to_next_bar_open_and_writes_api_manifest():
    bars = [
        Bar("AAPL", datetime(2024, 1, 1, tzinfo=UTC), 100, 100, 100, 100),
        Bar("AAPL", datetime(2024, 1, 2, tzinfo=UTC), 110, 110, 110, 110),
    ]

    result = run_backtest(bars, BuyFirstBar(), metadata={"experiment": "api-contract"})

    assert result.fills[0].timestamp == bars[1].timestamp
    assert result.fills[0].price == 110
    assert result.run_manifest["metadata"] == {"api_version": API_VERSION, "experiment": "api-contract"}


def test_stable_api_accepts_immutable_config_and_creates_fresh_brokers():
    config = BacktestConfig(broker=BrokerConfig(initial_cash=1_000, fill_price=FillPrice.OPEN))
    bars = [
        Bar("AAPL", datetime(2024, 1, 1, tzinfo=UTC), 10, 10, 10, 10),
        Bar("AAPL", datetime(2024, 1, 2, tzinfo=UTC), 11, 11, 11, 11),
    ]

    first = run_backtest(bars, BuyFirstBar(), config=config)
    second = run_backtest(bars, BuyFirstBar(), config=config)

    assert first.summary()["final_equity"] == second.summary()["final_equity"] == 1_000
    assert config.broker.initial_cash == 1_000


def test_strategy_spec_creates_fresh_instances_and_records_parameters():
    bars = [
        Bar("AAPL", datetime(2024, 1, 1, tzinfo=UTC), 10, 10, 10, 10),
        Bar("AAPL", datetime(2024, 1, 2, tzinfo=UTC), 11, 11, 11, 11),
    ]
    spec = StrategySpec(ParameterizedBuy, {"quantity": 2}, name="buy-two")

    first = run_backtest(bars, spec)
    second = run_backtest(bars, spec)

    assert first.fills[0].quantity == second.fills[0].quantity == 2
    assert first.run_manifest["metadata"]["strategy_spec"] == {
        "identity": "buy-two",
        "parameters": {"quantity": 2},
    }
