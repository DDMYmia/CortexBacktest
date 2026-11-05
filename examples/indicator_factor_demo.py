"""Build a technical factor and run cross-sectional factor research."""

from __future__ import annotations

import polars as pl

from cortexbacktest import momentum, run_factor_research
if __package__:
    from ._demo_data import parsed_business_dates as _business_days
else:
    from _demo_data import parsed_business_dates as _business_days



def main() -> None:
    dates = _business_days("2024-01-01", 40)
    prices = pl.DataFrame(
        {
            "date": dates,
            "AAPL": [100 + i * 0.8 for i in range(len(dates))],
            "MSFT": [100 + i * 0.4 for i in range(len(dates))],
            "XOM": [100 - i * 0.2 for i in range(len(dates))],
            "JPM": [100 + (-1) ** i * 0.3 + i * 0.1 for i in range(len(dates))],
        }
    )
    factor = momentum(prices, window=5)
    report = run_factor_research(prices, factor, horizons=(1, 5), quantiles=2)

    print("CortexBacktest - indicator factor demo")
    print(report.summary())
    print(f"coverage: {report.coverage_mean:.4f}")
    print(f"final_equity: {report.final_equity:.4f}")


if __name__ == "__main__":
    main()
