from cortexbacktest import FillPrice
from datetime import UTC, datetime, timedelta

from cortexbacktest import (
    BacktestEngine,
    Bar,
    BarGenerator,
    BrokerSimulator,
    Contract,
    CtaTemplate,
    HistoricalDataStore,
    PortfolioStrategyTemplate,
    Strategy,
    Tick,
    TickRecorder,
    TickReplayer,
    VnpyBacktestingConfig,
    crossover,
    crossunder,
    optimize_strategy,
    ticks_to_bars,
)


def _bars():
    return [
        Bar("AAPL", datetime(2024, 1, 1, tzinfo=UTC), 10, 10, 10, 10),
        Bar("AAPL", datetime(2024, 1, 2, tzinfo=UTC), 11, 11, 11, 11),
        Bar("AAPL", datetime(2024, 1, 3, tzinfo=UTC), 12, 12, 12, 12),
    ]


class ParamStrategy(Strategy):
    def __init__(self, quantity: int = 1, enter_day: int = 0):
        super().__init__()
        self.quantity = quantity
        self.enter_day = enter_day
        self.index = 0

    def on_bar(self, bar):
        if self.index == self.enter_day:
            self.buy(bar.symbol, quantity=self.quantity)
        self.index += 1
        return []


def test_optimize_strategy_returns_heatmap_and_best_params():
    result = optimize_strategy(
        _bars(),
        ParamStrategy,
        maximize="final_equity",
        quantity=[1, 2],
        enter_day=[0, 1],
    )

    heatmap = result.heatmap_pl()

    assert heatmap.height == 4
    assert result.best_params()["quantity"] == 2
    assert "final_equity" in heatmap.columns
    assert result.heatmap_matrix_pl(x="quantity", y="enter_day").height == 2
    assert "CortexBacktest Strategy Optimization" in result.report_html()


def test_optimize_strategy_random_samples_are_reproducible():
    first = optimize_strategy(
        _bars(),
        ParamStrategy,
        method="random",
        max_tries=2,
        random_state=7,
        quantity=[1, 2, 3],
        enter_day=[0, 1],
    )
    second = optimize_strategy(
        _bars(),
        ParamStrategy,
        method="random",
        max_tries=2,
        random_state=7,
        quantity=[1, 2, 3],
        enter_day=[0, 1],
    )

    assert first.rows == second.rows
    assert len(first.rows) == 2


def test_optimize_strategy_supports_constraint_and_custom_metric():
    result = optimize_strategy(
        _bars(),
        ParamStrategy,
        maximize=lambda summary: summary["fills"],
        constraint=lambda params: params["quantity"] == 1,
        quantity=[1, 2],
        enter_day=[0, 1],
    )

    assert result.metric == "score"
    assert result.heatmap_pl().height == 2


def test_optimize_strategy_thread_backend_matches_serial():
    serial = optimize_strategy(_bars(), ParamStrategy, parallel_backend="serial", quantity=[1, 2, 3], enter_day=[0, 1])
    threaded = optimize_strategy(
        _bars(), ParamStrategy, parallel_backend="thread", max_workers=2, min_cases_for_parallel=1, quantity=[1, 2, 3], enter_day=[0, 1]
    )
    assert sorted(threaded.rows, key=lambda row: row["case"]) == sorted(serial.rows, key=lambda row: row["case"])


def test_optimize_strategy_process_backend_matches_serial():
    # ParamStrategy must stay module-level: the process backend pickles it.
    serial = optimize_strategy(_bars(), ParamStrategy, parallel_backend="serial", quantity=[1, 2, 3], enter_day=[0, 1])
    parallel = optimize_strategy(
        _bars(), ParamStrategy, parallel_backend="process", max_workers=2, min_cases_for_parallel=1, quantity=[1, 2, 3], enter_day=[0, 1]
    )
    assert sorted(parallel.rows, key=lambda row: row["case"]) == sorted(serial.rows, key=lambda row: row["case"])


def test_optimize_strategy_auto_backend_stays_serial_below_min_cases():
    # Only 2 cases; min_cases_for_parallel defaults to 4, so "auto" must not
    # spin up a pool for them even though max_workers says it could.
    result = optimize_strategy(
        _bars(), ParamStrategy, parallel_backend="auto", max_workers=4, quantity=[1, 2], enter_day=[0]
    )
    assert len(result.rows) == 2


def test_optimize_strategy_early_stop_halts_the_search():
    seen = []

    def stop_after_two(row):
        seen.append(row["case"])
        return len(seen) >= 2

    result = optimize_strategy(
        _bars(),
        ParamStrategy,
        parallel_backend="serial",
        early_stop=stop_after_two,
        quantity=[1, 2, 3, 4, 5],
        enter_day=[0],
    )
    assert len(result.rows) == 2


def test_optimize_strategy_batches_a_large_grid_without_materializing_it():
    # 50 x 20 = 1,000 cases; batch_size=7 forces multiple batches through the
    # thread path, and max_tries=25 exercises islice's lazy truncation of the
    # underlying itertools.product rather than building all 1,000 first.
    result = optimize_strategy(
        _bars(),
        ParamStrategy,
        parallel_backend="thread",
        max_workers=3,
        min_cases_for_parallel=1,
        batch_size=7,
        max_tries=25,
        quantity=list(range(1, 51)),
        enter_day=[0] * 20,
    )
    assert len(result.rows) == 25


def test_optimize_strategy_random_reservoir_sample_has_no_duplicates_and_right_size():
    result = optimize_strategy(
        _bars(),
        ParamStrategy,
        method="random",
        max_tries=10,
        random_state=3,
        quantity=list(range(1, 21)),
        enter_day=[0, 1],
    )
    cases = [row["case"] for row in result.rows]
    assert len(cases) == 10
    assert len(set(cases)) == 10  # every sampled combination is distinct


def test_crossover_helpers():
    assert crossover([1, 2], [2, 1])
    assert crossunder([2, 1], [1, 2])
    assert crossover([1, 3], 2)


def test_vnpy_bar_generator_and_ticks_to_bars():
    start = datetime(2024, 1, 1, 9, 30, 1, tzinfo=UTC)
    ticks = [
        Tick("IF2401", start, 100, volume=1),
        Tick("IF2401", start + timedelta(seconds=20), 102, volume=2),
        Tick("IF2401", start + timedelta(minutes=1), 101, volume=3),
    ]
    bars = []
    generator = BarGenerator(bars.append)
    for tick in ticks:
        generator.update_tick(tick)
    generator.close()

    direct = ticks_to_bars(ticks)

    assert [(bar.open, bar.high, bar.low, bar.close, bar.volume) for bar in bars] == [
        (100, 102, 100, 102, 3),
        (101, 101, 101, 101, 3),
    ]
    assert direct == bars


def test_cta_template_lifecycle_maps_to_strategy_hooks():
    class DemoCta(CtaTemplate):
        def __init__(self):
            super().__init__()
            self.events = []
            self.done = False

        def on_init(self):
            self.events.append("init")

        def on_start(self):
            self.events.append("start")

        def on_stop(self):
            self.events.append("stop")

        def on_bar(self, bar):
            if not self.done:
                self.done = True
                self.buy(bar.symbol, quantity=1)
            return []

    strategy = DemoCta()
    result = BacktestEngine(trade_on_close=True).run(_bars(), strategy)

    assert strategy.events == ["init", "start", "stop"]
    assert result.fill_count == 1


def test_contract_and_vnpy_backtesting_config():
    contract = Contract("IF2401", exchange="CFFEX", size=300, price_tick=0.2)
    config = VnpyBacktestingConfig(contract=contract, capital=1_000_000, size=contract.size)

    assert config.contract.symbol == "IF2401"
    assert config.size == 300


def test_contract_multiplier_changes_pnl_and_cash():
    bars = [
        Bar("IF2401", datetime(2024, 1, 1, tzinfo=UTC), 100, 100, 100, 100),
        Bar("IF2401", datetime(2024, 1, 2, tzinfo=UTC), 101, 101, 101, 101),
    ]

    result = BacktestEngine(trade_on_close=True, 
        broker=BrokerSimulator(fill_price=FillPrice.CLOSE,
            initial_cash=100_000,
            contract_multipliers={"IF2401": 300},
        )
    ).run(bars, ParamStrategy())

    assert result.performance.equity_curve()["cash"][0] == 70_000
    assert result.performance.equity_curve()["equity"][-1] == 100_300


def test_historical_data_store_and_tick_replayer(tmp_path):
    store = HistoricalDataStore(tmp_path)
    ticks = [
        Tick("AAPL", datetime(2024, 1, 1, 9, 31, tzinfo=UTC), 11, volume=2),
        Tick("AAPL", datetime(2024, 1, 1, 9, 30, tzinfo=UTC), 10, volume=1),
    ]
    recorder = TickRecorder(store, "demo")
    for tick in ticks:
        recorder.record(tick)
    recorder.flush()

    replayed = list(TickReplayer(store, "demo").replay())
    bars = ticks_to_bars(replayed)
    store.write_bars("demo_bars", bars)

    assert [tick.last_price for tick in replayed] == [10, 11]
    assert store.list_datasets("ticks") == ["demo"]
    assert store.read_bars("demo_bars")[0].symbol == "AAPL"


def test_historical_data_store_preserves_futures_and_market_state_fields(tmp_path):
    store = HistoricalDataStore(tmp_path)
    bar = Bar(
        "IF2401",
        datetime(2024, 1, 1, tzinfo=UTC),
        100,
        100,
        100,
        100,
        open_interest=12_345,
        halted=True,
        limit_up=110,
        limit_down=90,
    )

    store.write_bars("if", [bar])

    assert store.read_bars("if") == [bar]


def test_portfolio_strategy_template_receives_market_slice():
    class DemoPortfolio(PortfolioStrategyTemplate):
        def __init__(self):
            super().__init__()
            self.events = []
            self.done = False

        def on_init(self):
            self.events.append("init")

        def on_start(self):
            self.events.append("start")

        def on_stop(self):
            self.events.append("stop")

        def on_portfolio_bars(self, bars):
            if not self.done:
                self.done = True
                self.buy("AAPL", quantity=1)
                self.buy("MSFT", quantity=1)
            return []

    bars = [
        Bar("AAPL", datetime(2024, 1, 1, tzinfo=UTC), 10, 10, 10, 10),
        Bar("MSFT", datetime(2024, 1, 1, tzinfo=UTC), 20, 20, 20, 20),
    ]
    strategy = DemoPortfolio()
    result = BacktestEngine(trade_on_close=True).run(bars, strategy)

    assert strategy.events == ["init", "start", "stop"]
    assert result.fill_count == 2
