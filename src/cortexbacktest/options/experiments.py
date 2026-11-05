"""Isolated execution scenarios and a strict tabular quote boundary."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import is_dataclass, replace
from typing import Callable, Iterable, Mapping, cast

import polars as pl

from .backtest import OptionBacktestConfig, OptionBacktestResult, OptionStrategy, run_option_backtest
from .contracts import OptionContract, OptionMarketFrame, OptionQuote, utc
from .costs import OptionCostModel


def run_option_scenarios(contracts: Iterable[OptionContract], frames: Iterable[OptionMarketFrame],
                         strategy_factory: Callable[[], OptionStrategy], *,
                         config: OptionBacktestConfig | None = None,
                         spread_fractions: tuple[float, ...] = (0, .25, .5, .75, 1),
                         workers: int = 1) -> dict[float, OptionBacktestResult]:
    """Fresh strategy/account per scenario; common fees, quotes, times and constraints.

    This comparison isolates spread assumptions, not latency or all friction.
    Thread mode preserves input order; factories/plugins must not share mutable state.
    """
    if (not spread_fractions or len(set(spread_fractions)) != len(spread_fractions)
            or isinstance(workers, bool) or not isinstance(workers, int) or workers < 1):
        raise ValueError("unique nonempty scenarios and positive integer workers required")
    config = config or OptionBacktestConfig()
    scenarios = []
    for value in spread_fractions:
        if config.costs is None:
            scenarios.append(replace(config, spread_fraction=value))
        else:
            plugin = config.costs
            if not is_dataclass(plugin) or isinstance(plugin, type) or not hasattr(plugin, "spread_fraction"):
                raise ValueError("a spread sweep needs a cost plugin with a spread_fraction field; "
                                 "vary the plugin yourself and call run_option_backtest directly")
            scenarios.append(replace(config, costs=cast(OptionCostModel, replace(plugin, spread_fraction=value))))
    contracts, frames = tuple(contracts), tuple(frames)
    # Construct sequentially, including mutable custom plugins, to avoid shared accounts.
    from copy import deepcopy
    tasks = [(deepcopy(cfg), strategy_factory()) for cfg in scenarios]
    def run(task):
        cfg, strategy = task
        return run_option_backtest(contracts, frames, strategy, config=cfg)
    if workers == 1:
        results = [run(task) for task in tasks]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(run, tasks))
    return dict(zip(spread_fractions, results))


def option_frames_from_polars(quotes: pl.DataFrame) -> tuple[OptionMarketFrame, ...]:
    """Strict quote-only adapter; timestamp is the processing/decision frame time.

    Required aware datetimes: timestamp, event_time, available_at. Optional
    features are assumed available_at; join their real publication times upstream.
    No expiry fixing is inferred: attach explicit lifecycle events before running.

    One `OptionQuote` per row has to exist, so this stage is O(rows) in Python
    objects however it is written. What it avoids is per-row overhead that is not
    the object: the frame is sorted once so each timestamp's quotes arrive
    contiguously, missing optional columns are filled in once rather than
    intersected per row, and rows come back as tuples that go straight into the
    constructor positionally.
    """
    required = ("timestamp", "symbol", "event_time", "available_at", "bid", "ask")
    missing = set(required) - set(quotes.columns)
    if missing:
        raise ValueError(f"missing quote columns: {sorted(missing)}")
    if any(quotes[name].null_count() for name in required):
        raise ValueError("null required quote fields")
    optional = ("bid_size", "ask_size", "implied_volatility", "delta", "open_interest")
    absent = [pl.lit(None, pl.Float64).alias(name)
              for name in optional if name not in quotes.columns]
    ordered = (quotes.with_columns(absent) if absent else quotes).select(
        "timestamp", "symbol", "event_time", "available_at", "bid", "ask", *optional
    ).sort("timestamp", maintain_order=True)
    frames: list[OptionMarketFrame] = []
    group: dict[str, OptionQuote] = {}
    now = None
    for row in ordered.iter_rows():
        if row[0] != now:
            if now is not None:
                frames.append(OptionMarketFrame(now, group))
            now, group = utc(row[0]), {}
        quote = OptionQuote(*row[1:])
        if quote.symbol in group:
            raise ValueError("duplicate symbol in quote frame")
        group[quote.symbol] = quote
    if now is not None:
        frames.append(OptionMarketFrame(now, group))
    return tuple(frames)


def option_scenario_summary(results: Mapping[float, OptionBacktestResult]) -> pl.DataFrame:
    return pl.DataFrame([{"spread_fraction": value, **result.summary(), **result.attribution}
                         for value, result in results.items()])
