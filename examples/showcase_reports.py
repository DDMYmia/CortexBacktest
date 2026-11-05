"""Write the HTML reports shown in the README from reproducible synthetic data.

    python examples/showcase_reports.py --output examples/output/showcase

A 40-symbol SMA(20/50) portfolio over three years of synthetic daily bars, with
5 bp commission and 2 bp slippage, filled at the next bar's open. Prices come from
a seeded random walk; nothing here is market evidence.
"""
from __future__ import annotations

import argparse
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np

from cortexbacktest.reporting import equity_curve_charts, write_report_html
from cortexbacktest import (
    BacktestEngine,
    Bar,
    BrokerSimulator,
    Order,
    Side,
    Strategy,
    run_factor_research,
    write_event_tear_sheet_html,
    write_research_report_html,
)

if __package__:
    from ._demo_data import business_dates
    from .factor_research_demo import build_demo_data
else:
    from _demo_data import business_dates
    from factor_research_demo import build_demo_data

SYMBOLS, DAYS, FAST, SLOW, SHARES = 40, 756, 20, 50, 50


def synthetic_bars() -> list[Bar]:
    rng = np.random.default_rng(11)
    days = business_dates(date(2023, 1, 2), DAYS)
    drift = rng.normal(0.0003, 0.0003, SYMBOLS)
    vol = rng.uniform(0.01, 0.025, SYMBOLS)
    close = 100.0 * np.exp(np.cumsum(drift + vol * rng.standard_normal((DAYS, SYMBOLS)), axis=0))
    gap = 1.0 + 0.004 * rng.standard_normal((DAYS, SYMBOLS))
    bars = []
    for i, day in enumerate(days):
        stamp = datetime(day.year, day.month, day.day, tzinfo=UTC)
        for s in range(SYMBOLS):
            c = float(close[i, s])
            o = float(close[i - 1, s] * gap[i, s]) if i else c
            bars.append(Bar(f"S{s:02d}", stamp, o, max(o, c) * 1.004, min(o, c) * 0.996, c, 1e6))
    return bars


class SmaCrossover(Strategy):
    def __init__(self) -> None:
        super().__init__()
        self.closes: dict[str, list[float]] = {}
        self.long: dict[str, bool] = {}

    def on_bar(self, bar: Bar):
        closes = self.closes.setdefault(bar.symbol, [])
        closes.append(bar.close)
        if len(closes) < SLOW:
            return []
        want = sum(closes[-FAST:]) / FAST > sum(closes[-SLOW:]) / SLOW
        if want == self.long.get(bar.symbol, False):
            return []
        self.long[bar.symbol] = want
        return [Order(bar.symbol, bar.timestamp, Side.BUY if want else Side.SELL, SHARES)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("examples/output/showcase"))
    args = parser.parse_args()
    broker = BrokerSimulator(initial_cash=250_000, commission_bps=5, slippage_bps=2, allow_short=False)
    result = BacktestEngine(broker=broker).run(synthetic_bars(), SmaCrossover())
    prices, factors = build_demo_data()
    research = run_factor_research(prices, factors, horizons=(1, 5, 20), quantiles=5,
                                   top_quantile=0.2, bottom_quantile=0.2, transaction_cost_bps=5.0)
    for path in (write_event_tear_sheet_html(result, args.output / "tear_sheet.html"),
                 write_report_html("SMA crossover, 40 symbols", result.report(include_records=False),
                                   args.output / "event_report.html",
                                   charts=equity_curve_charts(result.performance.equity_curve())),
                 write_research_report_html(research, args.output / "research_report.html")):
        print(path)
    summary = result.summary()
    print(f"final_equity {summary['final_equity']:,.2f}  fills {result.fill_count}")


if __name__ == "__main__":
    main()
