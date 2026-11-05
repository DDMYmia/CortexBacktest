"""Run factor research from plain bar and factor export files."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from cortexbacktest import build_research_bundle, evaluate_factor, simulate_target_weights
from cortexbacktest.research import top_bottom_weights
if __package__:
    from ._demo_data import business_date_strings as _business_days
else:
    from _demo_data import business_date_strings as _business_days


def write_demo_exports(output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(11)
    dates = _business_days("2024-01-02", 120)
    symbols = ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "AVGO", "TSLA"]
    quality = dict(zip(symbols, np.linspace(-1.0, 1.0, len(symbols))))

    prices = []
    factors = []
    last_prices = dict.fromkeys(symbols, 100.0)
    for day in dates:
        for symbol in symbols:
            daily_return = 0.0003 + quality[symbol] * 0.001 + rng.normal(0.0, 0.01)
            last_prices[symbol] = last_prices[symbol] * (1.0 + daily_return)
            prices.append(
                {
                    "date": day,
                    "symbol": symbol,
                    "close": round(last_prices[symbol], 4),
                    "adjClose": round(last_prices[symbol], 4),
                }
            )
            factors.append(
                {
                    "analysis_timestamp": day,
                    "ticker": symbol,
                    "smart_money_score": float(quality[symbol] + rng.normal(0.0, 0.5)),
                }
            )

    bars_path = output_dir / "fusion_bars.csv"
    factors_path = output_dir / "fusion_factors.csv"
    pl.DataFrame(prices).write_csv(bars_path)
    pl.DataFrame(factors).write_csv(factors_path)
    return bars_path, factors_path


def main() -> None:
    bars_path, factors_path = write_demo_exports(Path("examples/output"))
    bundle = build_research_bundle(
        bars_path,
        factors_path,
        forward_periods=5,
        factor_name="smart_money_score",
        research_mode="exploration",
    )
    result = evaluate_factor(bundle.research_frame, quantiles=4)
    weights = top_bottom_weights(
        bundle.factor_values,
        top_quantile=0.25,
        bottom_quantile=0.25,
    )
    equity = simulate_target_weights(bundle.returns, weights, transaction_cost_bps=5.0)

    print("CortexBacktest - export files demo")
    print(f"bars_file: {bars_path}")
    print(f"factors_file: {factors_path}")
    print(f"symbols: {bundle.prices.shape[1] - 1}")
    print(f"dates: {bundle.prices.shape[0]}")
    print(f"research_observations: {len(bundle.research_frame)}")
    print(f"mean_rank_ic: {result.mean_rank_ic:.4f}")
    print(f"mean_top_bottom_spread_5d: {result.mean_spread:.4f}")
    print(f"final_equity: {equity['equity'][-1]:.4f}")


if __name__ == "__main__":
    main()
