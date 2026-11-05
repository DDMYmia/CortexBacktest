"""Smoke test for an installed cortexbacktest, run outside the repo.

This exercises the package the way a user gets it: installed from a built
distribution into an environment holding only the declared dependencies. It
exists to catch undeclared dependencies, which the repo test suite cannot see
because a dev environment tends to have extra packages installed.

It caught a real one: Rank IC ran pandas' spearman path, which imports scipy,
while scipy was only in the optional reference-bench extra. A clean install
crashed on `evaluate_factor`.

Run it with the *installed* interpreter, from a directory that is not the repo
root, so `import cortexbacktest` resolves to the installed package
rather than the source tree:

    python -m venv /tmp/cleanenv
    /tmp/cleanenv/bin/python -m pip install dist/*.tar.gz
    cd /tmp && /tmp/cleanenv/bin/python <repo>/scripts/clean_install_smoke.py
"""

from __future__ import annotations

import importlib.util
from datetime import datetime, timedelta, timezone
from pathlib import Path
import subprocess
import sys

# Packages that must NOT be needed at runtime. They are optional extras, so if
# an import path starts depending on one the clean install must fail here.
FORBIDDEN_RUNTIME_PACKAGES = ("pandas", "scipy", "matplotlib", "backtrader", "alphalens")


def check_not_running_from_source_tree() -> None:
    import cortexbacktest

    package_dir = Path(cortexbacktest.__file__).resolve().parent
    # src layout: <repo>/src/cortexbacktest -> parents[1] is the repo root.
    # An installed copy lives under site-packages and has no pyproject.toml above it.
    if (package_dir.parents[1] / "pyproject.toml").exists():
        raise SystemExit(
            f"refusing to smoke test the source tree at {package_dir}; run this from "
            "another directory with the installed interpreter"
        )
    print(f"  package under test: {package_dir}")


def check_forbidden_packages_absent() -> None:
    present = [name for name in FORBIDDEN_RUNTIME_PACKAGES if importlib.util.find_spec(name) is not None]
    if present:
        raise SystemExit(
            f"optional-only packages are installed ({', '.join(present)}); this "
            "environment cannot prove the runtime dependency set is complete"
        )
    print(f"  confirmed absent: {', '.join(FORBIDDEN_RUNTIME_PACKAGES)}")


def check_event_backtest() -> None:
    from cortexbacktest import BacktestEngine, Bar, Order, Side, Strategy

    class BuyFirst(Strategy):
        def __init__(self) -> None:
            self.done = False

        def on_bars(self, bars: list[Bar]) -> list[Order]:
            if self.done:
                return []
            self.done = True
            return [Order(bar.symbol, bar.timestamp, Side.BUY, 1) for bar in bars]

    start = datetime(2024, 1, 1)
    bars = [
        Bar("AAA", start + timedelta(days=i), 100.0 + i, 101.0 + i, 99.0 + i, 100.5 + i)
        for i in range(10)
    ]
    # Manifest construction is opt-in in the current engine API.
    result = BacktestEngine(build_manifest=True).run(bars, BuyFirst())
    summary = result.summary()
    if summary["fills"] != 1:
        raise SystemExit(f"expected 1 fill, got {summary['fills']}")
    if result.performance.equity_curve().height != len(bars):
        raise SystemExit("equity curve height does not match bar count")
    if result.run_manifest is None:
        raise SystemExit("run manifest missing when explicitly requested")
    print(f"  event backtest OK (final_equity={summary['final_equity']:.2f}, fills={summary['fills']})")


def check_rank_ic() -> None:
    """Rank IC once required scipy via pandas' spearman; it must now be pandas-free."""
    import polars as pl

    from cortexbacktest.research.workflow import evaluate_factor

    dates = [f"2024-01-{day:02d}" for day in range(1, 13)]
    rows = [
        {"date": date, "symbol": symbol, "factor": float(rank), "forward_return": 0.01 * rank}
        for date in dates
        for rank, symbol in enumerate(("A", "B", "C", "D"))
    ]
    report = evaluate_factor(
        pl.DataFrame(rows),
        date_col="date",
        symbol_col="symbol",
        factor_col="factor",
        forward_return_col="forward_return",
    )
    rank_ic = float(report.rank_ic_by_date["rank_ic"].mean())
    if not rank_ic > 0.99:
        raise SystemExit(f"monotonic factor should give rank IC ~1.0, got {rank_ic}")
    print(f"  rank IC OK (mean={rank_ic:.4f}, pandas-free)")


def check_cli() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "cortexbacktest.cli", "--help"],
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise SystemExit(f"CLI --help failed: {completed.stderr}")
    print("  CLI entry point OK")


def check_numerical_options_and_risk() -> None:
    import polars as pl
    from cortexbacktest.options import black_scholes, crank_nicolson_price
    from cortexbacktest.reporting import var_backtest

    args = ("CALL", 100., 100., 1., .05, .2)
    actual = crank_nicolson_price(*args, american=False)
    if abs(actual - black_scholes(*args)) > .003:
        raise SystemExit("NumPy finite-difference option price disagrees with closed form")
    dates = [datetime(2025, 1, 2, tzinfo=timezone.utc) + timedelta(days=i) for i in range(3)]
    starts = [d - timedelta(days=1) for d in dates]
    returns = pl.DataFrame({"timestamp": dates, "period_start": starts, "return": [-.01, -.03, .01]})
    forecasts = pl.DataFrame({"timestamp": dates, "forecast_at": starts, "var": [.02] * 3})
    if var_backtest(returns, forecasts).summary["violations"] != 1:
        raise SystemExit("VaR exception calculation failed")
    print("  NumPy option pricing and risk calibration OK (no SciPy)")


def check_cashflow_returns() -> None:
    import polars as pl
    from cortexbacktest.reporting import cashflow_performance

    start = datetime(2023, 1, 1, tzinfo=timezone.utc)
    end = start + timedelta(days=730)
    nav = pl.DataFrame({"timestamp": [start, end], "value": [100., 121.], "currency": ["USD", "USD"]})
    flows = pl.DataFrame(schema={"timestamp": pl.Datetime("us", "UTC"), "flow_id": pl.String,
        "amount": pl.Float64, "currency": pl.String, "kind": pl.String})
    result = cashflow_performance(nav, flows, currency="USD", coverage_complete=True, as_of=end)
    if abs(result.summary["twr"]["totalReturn"] - .21) > 1e-10 or abs(result.summary["mwr"]["annualizedReturn"] - .1) > 1e-10:
        raise SystemExit("cash-flow return arithmetic failed")
    print("  TWR/MWR cash-flow reporting OK")


def check_group_attribution() -> None:
    import polars as pl
    from cortexbacktest.reporting import group_attribution

    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    end = start + timedelta(days=1)
    groups = pl.DataFrame({"timestamp": [end, end], "period_start": [start, start],
        "weights_at": [start, start], "classification_at": [start, start], "currency": ["USD"] * 2,
        "group": ["A", "B"], "portfolio_weight": [.6, .4], "benchmark_weight": [.5, .5],
        "portfolio_return": [.05, .004], "benchmark_return": [.03, .01]})
    observed = pl.DataFrame({"timestamp": [end], "period_start": [start], "currency": ["USD"],
        "portfolio_return": [.0316], "benchmark_return": [.02]})
    report = group_attribution(groups, observed, currency="USD", benchmark_id="smoke")
    if abs(report.summary["active_return"] - .0116) > 1e-12 or abs(report.summary["reconciliation_residual"]) > 1e-12:
        raise SystemExit("group attribution failed reconciliation")
    print("  Brinson/Carino attribution OK")


def check_factor_risk() -> None:
    import math
    import polars as pl
    from cortexbacktest.reporting import factor_risk_decomposition, portfolio_style_exposure

    stamp = datetime(2025, 1, 1, tzinfo=timezone.utc)
    metadata = {"timestamp": [stamp], "known_at": [stamp], "symbol": ["A"]}
    weights = pl.DataFrame({**metadata, "weight": [1.]})
    exposures = pl.DataFrame({**metadata, "market": [1.]})
    style = portfolio_style_exposure(weights.with_columns(pl.lit(1.).alias("benchmark_weight")),
        exposures, model_id="smoke-market-only")
    if style.history["active_exposure"][0] != 0:
        raise SystemExit("active style exposure failed")
    risk = factor_risk_decomposition(weights, exposures,
        pl.DataFrame({"factor": ["market"], "market": [.04]}),
        pl.DataFrame({**metadata, "variance": [.01]}), as_of=stamp, model_known_at=stamp,
        fit_end=stamp, model_id="smoke-market-only", currency="USD", horizon="daily")
    if abs(risk["totalVariance"] - .05) > 1e-12 or abs(risk["volatility"] - math.sqrt(.05)) > 1e-12:
        raise SystemExit("factor risk failed")
    print("  Named style exposure and factor risk OK")


def check_holding_recipe() -> None:
    import json
    from cortexbacktest.research import HoldingConfig, PersistentWeightRecipe

    stamp = datetime(2025, 1, 1, tzinfo=timezone.utc)
    config = HoldingConfig(enter_threshold=.5, exit_threshold=-.5,
        enter_periods=2, cooldown_periods=1, holding_periods=2)
    recipe = PersistentWeightRecipe(["A"], config=config)
    if recipe.update(stamp, {"A": 1.}, {"A": 1.})["A"] != 0:
        raise SystemExit("signal entered before persistence filled")
    restored = PersistentWeightRecipe.from_document(json.loads(json.dumps(recipe.to_document())),
        symbols=["A"], config=config)
    if restored.update(stamp + timedelta(days=1), {"A": 1.}, {"A": 1.})["A"] != .5:
        raise SystemExit("cohort capital or restored persistence failed")
    if restored.update(stamp + timedelta(days=2), {"A": None}, {"A": 1.})["A"] != 0:
        raise SystemExit("missing input retained an active cohort")
    print("  Persistence, cooldown checkpoint and funded cohorts OK")


def check_external_indicator_gate() -> None:
    import polars as pl
    from cortexbacktest.research.indicator_registry import IndicatorRegistry

    registry = IndicatorRegistry()
    data = pl.DataFrame({"date": list(range(8)), "A": [10. + i for i in range(8)]})
    requirements = registry.describe("tulip:sma", period=3)
    if requirements["warmup_bars"] != 3:
        raise SystemExit("external provider metadata unavailable without optional packages")
    for name in ["tulip:sma", "pandas_ta:sma"]:
        if registry.assess(name, data)["provider_available"]:
            raise SystemExit("clean gate check requires optional providers absent")
        try:
            registry.run(name, data)
        except ImportError:
            pass
        else:
            raise SystemExit("missing external provider unexpectedly computed a factor")
    if registry.describe("pandas_ta:vwap")["requirements_known"]:
        raise SystemExit("unknown provider warmup was guessed")
    print("  External metadata and missing-provider refusal OK")


def check_alpha_diagnostics_contract() -> None:
    from cortexbacktest.research.alpha_diagnostics import ALPHA_NAMES
    from cortexbacktest.adapters.qlib_predictions import _VALIDATOR

    if len(ALPHA_NAMES) != 157 or len(set(ALPHA_NAMES)) != 157:
        raise SystemExit("Alpha157 feature contract invalid")
    files = _VALIDATOR.schema["properties"]["files"]
    if set(files["dependentRequired"]["features.parquet"]) != {"diagnostics.json"}:
        raise SystemExit("Alpha diagnostic file binding missing")
    if any(importlib.util.find_spec(p) is not None for p in ("qlib", "lightgbm")):
        raise SystemExit("Alpha consumer clean check requires Qlib and LightGBM absent")
    print("  Alpha157 diagnostic consumer imports without Qlib/LightGBM")


def check_research_recipes() -> None:
    import polars as pl
    from cortexbacktest.research.recipes import group_budget_weights
    from cortexbacktest.research.pairs import PairConfig, PairModel, PairRecipe
    stamp = datetime(2020, 1, 1, tzinfo=timezone.utc)
    model = PairModel("A", "B", 0., 1., 0., 1., 0., stamp, stamp, stamp, "0"*64)
    recipe = PairRecipe((model,), config=PairConfig(max_pairs=1))
    weights = recipe.update(stamp+timedelta(days=1), {"A": 10., "B": 1.})
    if weights != {"A": -.5, "B": .5}:
        raise SystemExit("pair hedge targets invalid")
    scores = pl.DataFrame({"date": [stamp], "A": [1.], "B": [-1.]})
    members = pl.DataFrame({"date": [stamp]*2, "known_at": [stamp]*2, "symbol": ["A", "B"], "group": ["X"]*2})
    if group_budget_weights(scores, members, {"X": .6}).values["A"][0] != .6:
        raise SystemExit("group cash budget invalid")
    from cortexbacktest.research.recipes import ResidualMomentumConfig, residual_momentum
    dates = [stamp+timedelta(days=i) for i in range(9)]
    known = list(dates)
    known[4] += timedelta(days=100)
    returns = pl.DataFrame({"date":dates,"known_at":known,"A":[1.,2.,3.,5.,6.,4.,8.,9.,7.]})
    factors = pl.DataFrame({"date":dates,"known_at":dates,"F":list(range(9))})
    check = residual_momentum(returns,factors,config=ResidualMomentumConfig(3,2,0),return_basis="excess")
    if check.diagnostics["reason"][4] != "asset_not_available_by_decision":
        raise SystemExit("late asset receipt bypassed residual gate")
    print("  Frozen pair and dated group recipes import without statistical extras")


def check_term_structure() -> None:
    import polars as pl
    from cortexbacktest.research.term_structure import term_structure
    stamp = datetime(2025,1,1,tzinfo=timezone.utc)
    rows = [dict(date=stamp,root="X",symbol=s,expiry=stamp+timedelta(days=d),
                 last_trade_at=stamp+timedelta(days=d),quote_at=stamp,known_at=stamp,
                 price=v,price_scale=1.,currency="USD",price_unit="UNIT",multiplier=1.)
            for s,d,v in [("F1",30,100.),("F2",60,110.)]]
    result = term_structure(pl.DataFrame(rows),max_quote_age=timedelta(0),annualized=False)
    if abs(result.values["X"][0] - (100/110-1)) > 1e-12:
        raise SystemExit("term curve ratio invalid")
    print("  Synchronous raw-contract curve without optional dependencies OK")


def main() -> int:
    print("clean-install smoke test")
    check_not_running_from_source_tree()
    check_forbidden_packages_absent()
    check_event_backtest()
    check_rank_ic()
    check_numerical_options_and_risk()
    check_cashflow_returns()
    check_group_attribution()
    check_factor_risk()
    check_holding_recipe()
    check_external_indicator_gate()
    check_alpha_diagnostics_contract()
    check_research_recipes()
    check_term_structure()
    check_cli()
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
