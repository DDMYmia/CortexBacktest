"""Synthetic plumbing demonstration, NOT historical evidence of profitability.

Run: .venv/bin/python examples/option_research_demo.py --output /tmp/options-demo
"""
from argparse import ArgumentParser
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

from cortexbacktest.options import (
    OptionBacktestConfig, OptionContract, OptionLifecycleEvent, OptionMarketFrame,
    OptionQuote, ScheduledOptionTargets, black_scholes, bsm_greeks,
    option_scenario_summary, run_option_scenarios,
)


def build_demo():
    start = datetime(2026, 1, 5, 21, tzinfo=timezone.utc)
    expiry = start + timedelta(days=30)
    contracts = tuple(OptionContract(right.upper(), "ETF", right, 100, expiry)
                      for right in ("CALL", "PUT"))
    frames = []
    # Synthetic deterministic path; option mids manufactured from BSM for testing.
    # The engine still fills only supplied bid/ask, never its own theoretical price.
    for day in range(31):
        at = start + timedelta(days=day)
        spot = 100 + day * .1 + (day % 5 - 2) * .2
        quotes = {"ETF": OptionQuote("ETF", at, at, spot-.02, spot+.02, 10000, 10000)}
        events = []
        for contract in contracts:
            if day < 30:
                tau = (30-day)/365
                value = black_scholes(contract.right, spot, 100, tau, .03, .25)
                greeks = bsm_greeks(contract.right, spot, 100, tau, .03, .25)
                quotes[contract.symbol] = OptionQuote(contract.symbol, at, at,
                    max(.001, value-.05), value+.05, 100, 100,
                    implied_volatility=.25, delta=greeks.delta)
            else:
                events.append(OptionLifecycleEvent(contract.symbol, at, at, spot))
        frames.append(OptionMarketFrame(at, quotes, tuple(events)))
    # Extra observation closes any remaining stock hedge after expiry.
    at = expiry + timedelta(days=1)
    frames.append(OptionMarketFrame(at, {"ETF": OptionQuote("ETF", at, at, 102.58, 102.62,10000,10000)}))
    template = ScheduledOptionTargets(contracts,
        {start: {"CALL": -2, "PUT": -2}}, hedge_every=timedelta(days=1))
    return contracts, frames, template


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("/tmp/fusion-options-demo"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    contracts, frames, template = build_demo()
    config = OptionBacktestConfig(debit_rate=.06, credit_rate=.02, stock_fee_bps=1,
                                   stock_borrow_rate=.01, data_id="synthetic-bsm-v1",
                                   strategy_id="short-straddle-daily-held-delta-v1")
    scenarios = run_option_scenarios(contracts, frames, template.strategy, config=config,workers=2)
    summary = option_scenario_summary(scenarios)
    summary.write_csv(args.output / "scenarios.csv")
    quote_rows = [{"timestamp": f.timestamp, **asdict(q)} for f in frames for q in f.quotes.values()]
    import polars as pl
    pl.DataFrame(quote_rows).write_parquet(args.output / "synthetic-quotes.parquet")
    for spread, result in scenarios.items():
        prefix = args.output / f"spread-{spread:g}"
        result.snapshots.write_csv(Path(str(prefix) + "-equity.csv"))
        result.fills.write_csv(Path(str(prefix) + "-fills.csv"))
        # Explicit filename suffix avoids collisions between decimal scenario names.
        (args.output / f"spread-{spread:g}-manifest.json").write_text(
            json.dumps(dict(result.manifest), indent=2), encoding="utf-8")
    print(summary.select("spread_fraction", "net_pnl", "option_spread", "stock_spread", "fills"))
    print(f"Synthetic evidence only: {args.output}")


if __name__ == "__main__":
    main()
