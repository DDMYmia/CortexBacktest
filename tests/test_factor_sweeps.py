import numpy as np
import polars as pl
import pytest

from cortexbacktest import (
    FactorPreprocessConfig,
    FactorSweepCase,
    factor_stability_report,
    momentum,
    run_factor_research,
    run_factor_sweep,
)


def _prices() -> pl.DataFrame:
    dates = [f"2024-01-{day:02d}" for day in (1, 2, 3, 4, 5, 8, 9, 10)]
    return pl.DataFrame(
        {
            "date": dates,
            "A": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0],
            "B": [10.0, 9.5, 9.0, 8.5, 8.0, 7.5, 7.0, 6.5],
            "C": [10.0, 10.2, 10.4, 10.6, 10.8, 11.0, 11.2, 11.4],
        }
    )


def _momentum_cases() -> list[FactorSweepCase]:
    return [
        FactorSweepCase("mom_1", lambda frame: momentum(frame, window=1), params={"window": 1}),
        FactorSweepCase(
            "mom_2",
            lambda frame: momentum(frame, window=2),
            params={"window": 2},
            preprocessing=FactorPreprocessConfig(zscore=True),
        ),
    ]


def test_run_factor_sweep_summarizes_multiple_cases():
    result = run_factor_sweep(_prices(), _momentum_cases(), horizons=(1, 2), quantiles=3)

    summary = result.summary()
    stability = result.stability()
    statistical = result.statistical_summary_pl()

    assert set(result.reports) == {"mom_1", "mom_2"}
    assert isinstance(summary, pl.DataFrame)
    assert isinstance(stability, pl.DataFrame)
    assert {"case", "horizon", "mean_rank_ic", "param_window"}.issubset(summary.columns)
    assert {"case", "avg_rank_ic", "positive_rank_ic_ratio", "final_equity"}.issubset(stability.columns)
    assert summary.filter((pl.col("case") == "mom_1") & (pl.col("horizon") == 1)).height == 1
    assert result.best_case() in {"mom_1", "mom_2"}
    assert {"case", "sharpe", "deflated_sharpe_ratio"}.issubset(statistical.columns)


def test_factor_stability_report_summarizes_single_report():
    prices = _prices()
    report = run_factor_research(prices, momentum(prices, window=1), horizons=(1, 2), quantiles=3)

    stability = factor_stability_report(report)

    row = stability.row(0, named=True)
    assert row["horizon_count"] == 2
    assert row["positive_rank_ic_ratio"] == pytest.approx(1.0)


def test_statistical_confidence_is_independent_of_display_annualization():
    result = run_factor_sweep(_prices(), _momentum_cases(), horizons=(1,), quantiles=3)
    daily = result.statistical_summary_pl(periods_per_year=252).sort("case")
    annual = result.statistical_summary_pl(periods_per_year=1).sort("case")

    assert daily["deflated_sharpe_ratio"].to_list() == pytest.approx(
        annual["deflated_sharpe_ratio"].to_list())
    assert daily["sharpe"].to_list() == pytest.approx(
        (annual["sharpe"] * 252**0.5).to_list())
    assert any(0 < value < 1 for value in annual["deflated_sharpe_ratio"])


def test_factor_sweep_rejects_duplicate_case_names():
    cases = [
        FactorSweepCase("dup", lambda frame: momentum(frame, window=1)),
        FactorSweepCase("dup", lambda frame: momentum(frame, window=2)),
    ]

    with pytest.raises(ValueError, match="Duplicate"):
        run_factor_sweep(_prices(), cases, horizons=(1,), quantiles=3)


def test_factor_sweep_rejects_empty_cases():
    with pytest.raises(ValueError, match="cases"):
        run_factor_sweep(_prices(), [], horizons=(1,), quantiles=3)


def test_factor_sweep_best_case_returns_none_when_metric_all_nan():
    prices = pl.DataFrame({"date": ["2024-01-01", "2024-01-02"], "A": [10.0, 11.0], "B": [20.0, 21.0]})

    result = run_factor_sweep(
        prices,
        [FactorSweepCase("mom_2", lambda frame: momentum(frame, window=2))],
        horizons=(1,),
        quantiles=2,
    )

    assert result.best_case() is None


def _random_walk_prices() -> pl.DataFrame:
    rng = np.random.default_rng(5)
    dates = _business_days(120)
    symbols = ["A", "B", "C", "D", "E", "F"]
    columns = {symbol: 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(dates)))) for symbol in symbols}
    return pl.DataFrame({"date": dates, **columns})


def _business_days(count: int) -> list[str]:
    days = pl.date_range(
        pl.date(2024, 1, 1), pl.date(2025, 12, 31), interval="1d", eager=True
    )
    business = days.filter(days.dt.weekday() <= 5)
    return [value.isoformat() for value in business.to_list()[:count]]


def _oracle_factor(frame: pl.DataFrame) -> pl.DataFrame:
    # Deliberate lookahead oracle for inference tests; never a tradable signal.
    symbols = [column for column in frame.columns if column != "date"]
    return frame.select(
        pl.col("date"),
        *[(pl.col(symbol).shift(-1) / pl.col(symbol) - 1).alias(symbol) for symbol in symbols],
    )


def _noise_factor(seed: int):
    def make(frame: pl.DataFrame) -> pl.DataFrame:
        rng = np.random.default_rng(seed)
        symbols = [column for column in frame.columns if column != "date"]
        return pl.DataFrame({"date": frame["date"], **{symbol: rng.normal(0, 1, frame.height) for symbol in symbols}})

    return make


def test_fdr_summary_keeps_real_signal_and_rejects_noise_that_passes_raw_alpha():
    prices = _random_walk_prices()
    cases = [FactorSweepCase("oracle", _oracle_factor)]
    cases += [FactorSweepCase(f"noise_{index}", _noise_factor(100 + index)) for index in range(9)]

    result = run_factor_sweep(prices, cases, horizons=(1,), quantiles=3)
    fdr = result.fdr_summary_pl(alpha=0.05)

    assert fdr.height == len(cases)
    assert set(fdr.columns) >= {"case", "horizon", "mean_rank_ic", "observations", "t_stat", "p_value", "q_value", "significant"}

    significant = set(fdr.filter(pl.col("significant"))["case"].to_list())
    raw_significant = set(fdr.filter(pl.col("p_value") <= 0.05)["case"].to_list())

    assert significant == {"oracle"}
    # With ten cases some pure noise clears a raw 5% threshold; FDR is what
    # stops those from being reported as discoveries.
    assert raw_significant > significant


def test_fdr_summary_q_values_never_below_p_values_and_sorts_best_first():
    prices = _random_walk_prices()
    cases = [FactorSweepCase(f"noise_{index}", _noise_factor(200 + index)) for index in range(5)]

    fdr = run_factor_sweep(prices, cases, horizons=(1,), quantiles=3).fdr_summary_pl()

    assert (fdr["q_value"] >= fdr["p_value"]).all()
    assert fdr["q_value"].to_list() == sorted(fdr["q_value"].to_list())


def test_fdr_summary_rejects_unknown_metric():
    prices = _random_walk_prices()
    result = run_factor_sweep(prices, [FactorSweepCase("noise", _noise_factor(1))], horizons=(1,), quantiles=3)

    with pytest.raises(ValueError, match="metric must be"):
        result.fdr_summary_pl(metric="sharpe")
