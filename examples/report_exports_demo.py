"""Export research, factor sweep, and execution sweep reports."""

from __future__ import annotations

import polars as pl

if __package__:
    from ._demo_data import business_date_strings as _business_days, flat_bars as build_bars
else:
    from _demo_data import business_date_strings as _business_days, flat_bars as build_bars



from cortexbacktest import (
    FactorSweepCase,
    RebalanceConfig,
    momentum,
    run_execution_sweep,
    run_factor_research,
    run_factor_sweep,
    write_execution_sweep_report_html,
    write_execution_sweep_report_json,
    write_factor_sweep_report_html,
    write_factor_sweep_report_json,
    write_research_report_html,
    write_research_report_json,
)

def build_prices() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "date": _business_days("2024-01-02", 4),
            "AAPL": [100.0, 102.0, 103.0, 104.0],
            "MSFT": [100.0, 99.0, 98.0, 97.0],
            "NVDA": [100.0, 101.0, 104.0, 108.0],
        }
    )


def main() -> None:
    prices = build_prices()
    factors = prices.select(
        pl.col("date"), pl.lit(1.0).alias("AAPL"), pl.lit(-1.0).alias("MSFT"), pl.lit(2.0).alias("NVDA")
    )
    research = run_factor_research(prices, factors, horizons=(1,), quantiles=3)
    sweep = run_factor_sweep(
        prices,
        [FactorSweepCase("mom_1", lambda frame: momentum(frame, window=1), top_quantile=1 / 3, bottom_quantile=None)],
        horizons=(1,),
        quantiles=3,
    )
    execution = run_execution_sweep(
        sweep,
        build_bars(prices),
        rebalance_config=RebalanceConfig(allow_fractional=False, min_trade_value=1.0),
    )

    output_dir = "examples/output"
    paths = [
        write_research_report_json(research, f"{output_dir}/research_report.json"),
        write_research_report_html(research, f"{output_dir}/research_report.html"),
        write_factor_sweep_report_json(sweep, f"{output_dir}/factor_sweep_report.json"),
        write_factor_sweep_report_html(sweep, f"{output_dir}/factor_sweep_report.html"),
        write_execution_sweep_report_json(execution, f"{output_dir}/execution_sweep_report.json"),
        write_execution_sweep_report_html(execution, f"{output_dir}/execution_sweep_report.html"),
    ]

    print("CortexBacktest - report exports demo")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
