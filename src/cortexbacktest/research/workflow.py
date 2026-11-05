"""Cross-sectional factor research utilities."""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from ..data import Universe, apply_universe_mask
from ..data.backend import normalize_date_column, to_backend
from .frames import (
    _to_polars_matrix,
    forward_returns_matrix,
    returns_matrix,
    stack_factor_and_forward_returns,
)
from .portfolio import simulate_target_weights
from .preprocessing import FactorPreprocessConfig, prepare_factor
from .preprocessing import _to_long, _to_wide as _factor_wide_form
from .diagnostics import cumulative_quantile_returns, factor_decay_summary, ic_summary


@dataclass(frozen=True, slots=True)
class FactorResearchResult:
    ic_by_date: pl.DataFrame
    rank_ic_by_date: pl.DataFrame
    quantile_returns: pl.DataFrame
    spread_by_date: pl.DataFrame

    @property
    def mean_ic(self) -> float:
        return _mean_polars_column(self.ic_by_date, "ic")

    @property
    def mean_rank_ic(self) -> float:
        return _mean_polars_column(self.rank_ic_by_date, "rank_ic")

    @property
    def mean_spread(self) -> float:
        return _mean_polars_column(self.spread_by_date, "spread")

    def ic_summary(self) -> pl.DataFrame:
        return ic_summary(self.ic_by_date, value_col="ic")

    def rank_ic_summary(self) -> pl.DataFrame:
        return ic_summary(self.rank_ic_by_date, value_col="rank_ic")

    def cumulative_quantile_returns(self) -> pl.DataFrame:
        return cumulative_quantile_returns(self.quantile_returns)


@dataclass(frozen=True, slots=True)
class FactorResearchReport:
    horizon_results: dict[int, FactorResearchResult]
    coverage_by_date: pl.DataFrame
    turnover_by_date: pl.DataFrame
    target_weights: pl.DataFrame
    equity_curve: pl.DataFrame

    def summary(self) -> pl.DataFrame:
        rows = []
        for horizon, result in self.horizon_results.items():
            rows.append(
                {
                    "horizon": horizon,
                    "mean_ic": result.mean_ic,
                    "mean_rank_ic": result.mean_rank_ic,
                    "mean_spread": result.mean_spread,
                }
            )
        if not rows:
            return _empty_polars(["horizon", "mean_ic", "mean_rank_ic", "mean_spread"])
        return pl.DataFrame(rows).sort("horizon")

    def coverage(self) -> pl.DataFrame:
        return self.coverage_by_date

    def turnover(self) -> pl.DataFrame:
        return self.turnover_by_date

    def target_weights_frame(self) -> pl.DataFrame:
        return self.target_weights

    def equity_curve_frame(self) -> pl.DataFrame:
        return self.equity_curve

    def factor_decay(self) -> pl.DataFrame:
        return factor_decay_summary(self)

    @property
    def coverage_mean(self) -> float:
        return _mean_polars_column(self.coverage_by_date, "coverage")

    @property
    def turnover_mean(self) -> float:
        return _mean_polars_column(self.turnover_by_date, "turnover")

    @property
    def final_equity(self) -> float:
        if self.equity_curve.is_empty() or "equity" not in self.equity_curve.columns:
            return float("nan")
        value = self.equity_curve["equity"][-1]
        return float(value) if value is not None else float("nan")


def evaluate_factor(
    frame,
    *,
    date_col: str = "date",
    symbol_col: str = "symbol",
    factor_col: str = "factor",
    forward_return_col: str = "forward_return",
    quantiles: int = 5,
) -> FactorResearchResult:
    """Evaluate a cross-sectional factor using Polars long-form data."""
    if quantiles < 2:
        raise ValueError("quantiles must be >= 2")
    data = _factor_eval_frame_pl(
        frame,
        date_col=date_col,
        symbol_col=symbol_col,
        factor_col=factor_col,
        forward_return_col=forward_return_col,
    )
    usable = (
        data.group_by("date").agg(
            pl.len().alias("_count"),
            pl.col("factor").n_unique().alias("_factor_unique"),
            pl.col("forward_return").n_unique().alias("_forward_unique"),
            pl.corr("factor", "forward_return").alias("ic"),
            pl.corr("factor", "forward_return", method="spearman").alias("rank_ic"),
        )
        .filter((pl.col("_count") >= 2) & (pl.col("_factor_unique") >= 2) & (pl.col("_forward_unique") >= 2))
        .sort("date")
    )
    if usable.is_empty():
        return FactorResearchResult(
            _empty_polars(["date", "ic"]),
            _empty_polars(["date", "rank_ic"]),
            _empty_polars(["date"]),
            _empty_polars(["date", "spread"]),
        )
    ic_by_date = usable.select("date", "ic").drop_nulls("ic")
    rank_ic_by_date = usable.select("date", "rank_ic").drop_nulls("rank_ic")
    usable_data = data.join(usable.select("date", "_factor_unique"), on="date", how="inner")
    # Assign each row to a factor quantile within its date using window
    # expressions instead of a per-group Python callback. Dates whose factor has
    # fewer distinct values than `quantiles` cannot be split and are dropped.
    count = pl.len().over("date")
    ordinal = pl.col("factor").rank(method="ordinal").over("date") - 1
    quantile = ((ordinal * quantiles / count).floor() + 1).cast(pl.Int64)
    tagged = usable_data.with_columns(
        quantile.alias("quantile"),
    ).filter(pl.col("_factor_unique") >= quantiles)
    quantile_long = (
        tagged.group_by(["date", "quantile"]).agg(pl.col("forward_return").mean())
        if not tagged.is_empty()
        else _empty_polars(["date", "quantile", "forward_return"])
    )
    if quantile_long.is_empty():
        quantile_returns = _empty_polars(["date"])
        spread_by_date = _empty_polars(["date", "spread"])
    else:
        quantile_returns = (
            quantile_long.pivot(
                index="date",
                on="quantile",
                values="forward_return",
                aggregate_function="first",
            )
            .sort("date")
            # Quantile buckets are integers rendered as column names, so they must
            # be ordered numerically. Lexicographic ordering would place "10"
            # between "1" and "2" and make the spread Q9 - Q1 for quantiles >= 10.
        )
        value_columns = sorted((column for column in quantile_returns.columns if column != "date"), key=int)
        quantile_returns = quantile_returns.select("date", *value_columns)
        spread_by_date = quantile_returns.select(
            [
                pl.col("date"),
                (pl.col(value_columns[-1]) - pl.col(value_columns[0])).alias("spread"),
            ]
        )
    return FactorResearchResult(ic_by_date, rank_ic_by_date, quantile_returns, spread_by_date)


def run_factor_research(
    prices: pl.DataFrame,
    factors: pl.DataFrame,
    *,
    universe: Universe | pl.DataFrame | None = None,
    preprocessing: FactorPreprocessConfig | None = None,
    horizons: tuple[int, ...] = (1, 5, 20),
    quantiles: int = 5,
    top_quantile: float = 0.2,
    bottom_quantile: float | None = 0.2,
    transaction_cost_bps: float = 0.0,
    decision_times: pl.DataFrame | None = None,
) -> FactorResearchReport:
    """Run a Polars-native matrix-based cross-sectional factor workflow.

    With ``decision_times`` (factor date, decision_time), prices must be an
    explicit executable-price grid, e.g. raw opening prints at their actual
    instants. Signals take effect strictly after the decision, carrying forward
    until another decision supersedes them. Labels and NAV then use executable
    price-to-price returns; output dates are execution instants. This does not
    certify input price vintages, liquidity, or historical universe membership.
    """
    if not horizons:
        raise ValueError("horizons must not be empty")
    if decision_times is not None:
        prices, factors = _execution_factors(prices, factors, decision_times)
    aligned_prices, aligned_factors = _align_wide_polars_matrices(prices, factors)
    aligned_prices = apply_universe_mask(aligned_prices, universe)
    aligned_factors = apply_universe_mask(aligned_factors, universe)
    if preprocessing is not None:
        aligned_factors = prepare_factor(aligned_factors, preprocessing)
    horizon_results = {}
    for horizon in horizons:
        forward_returns = forward_returns_matrix(aligned_prices, periods=horizon)
        research_frame = stack_factor_and_forward_returns(aligned_factors, forward_returns)
        horizon_results[horizon] = evaluate_factor(research_frame, quantiles=quantiles)
    target_weights = top_bottom_weights(
        aligned_factors,
        top_quantile=top_quantile,
        bottom_quantile=bottom_quantile,
    )
    equity_curve = simulate_target_weights(
        returns_matrix(aligned_prices),
        target_weights,
        transaction_cost_bps=transaction_cost_bps,
    )
    return FactorResearchReport(
        horizon_results=horizon_results,
        coverage_by_date=_factor_coverage_pl(aligned_factors),
        turnover_by_date=_weight_turnover_pl(target_weights),
        target_weights=target_weights,
        equity_curve=equity_curve,
    )


def _execution_factors(prices: pl.DataFrame, factors: pl.DataFrame, decisions: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    from .revisions import _clocks

    prices = _clocks(prices, ("date",)).sort("date")
    grid = prices.select("date")
    factors = _clocks(factors, ("date",)).sort("date")
    decisions = _clocks(decisions, ("date", "decision_time")).sort("date")
    if (grid.is_empty() or grid["date"].n_unique() != grid.height
            or factors["date"].n_unique() != factors.height
            or not factors["date"].equals(decisions["date"])):
        raise ValueError("Execution prices require unique dates and every factor row needs one decision")
    if (decisions.filter(pl.col("decision_time") < pl.col("date")).height
            or not decisions["decision_time"].is_sorted()
            or decisions["decision_time"].n_unique() != decisions.height):
        raise ValueError("Decision times must be unique, chronological and no earlier than factor dates")
    if set(prices.columns) != set(factors.columns):
        raise ValueError("Execution prices and factors must have the same asset columns")
    if prices.select(((pl.exclude("date") <= 0) | ~pl.exclude("date").is_finite()).any()).row(0).count(True):
        raise ValueError("Observed execution prices must be finite and positive")
    timed = factors.join(decisions, on="date").drop("date").sort("decision_time")
    # Equality gives no time to act on the decision; wait for the next print.
    # A new null factor replaces the previous signal, never forward-fills it.
    return prices, grid.join_asof(timed, left_on="date", right_on="decision_time",
                                 strategy="backward", allow_exact_matches=False).drop("decision_time")


def _factor_eval_frame_pl(
    frame,
    *,
    date_col: str,
    symbol_col: str,
    factor_col: str,
    forward_return_col: str,
) -> pl.DataFrame:
    data = to_backend(frame, backend="polars")
    required = {date_col, symbol_col, factor_col, forward_return_col}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    data = normalize_date_column(data, date_col)
    return (
        data.select(
            [
                pl.col(date_col).alias("date"),
                pl.col(symbol_col).cast(pl.Utf8).alias("symbol"),
                pl.col(factor_col).cast(pl.Float64, strict=False).alias("factor"),
                pl.col(forward_return_col).cast(pl.Float64, strict=False).alias("forward_return"),
            ]
        )
        .drop_nulls(["date", "symbol", "factor", "forward_return"])
        .filter(pl.col("factor").is_finite() & pl.col("forward_return").is_finite())
        .sort(["date", "symbol"])
    )




def _mean_polars_column(frame: pl.DataFrame, column: str) -> float:
    if frame.is_empty() or column not in frame.columns:
        return float("nan")
    value = frame[column].mean()
    return float(value) if value is not None else float("nan")


def _empty_polars(columns: list[str]) -> pl.DataFrame:
    return pl.DataFrame({column: [] for column in columns})


def _align_wide_polars_matrices(prices: pl.DataFrame, factors) -> tuple[pl.DataFrame, pl.DataFrame]:
    price_frame = _wide_matrix_pl(prices)
    factor_frame = _wide_matrix_pl(factors)
    symbols = sorted(set(price_frame.columns).intersection(factor_frame.columns).difference({"date"}))
    common_dates = price_frame.select("date").join(factor_frame.select("date"), on="date", how="inner").unique().sort("date")
    return (
        price_frame.join(common_dates, on="date", how="inner").select(["date", *symbols]).sort("date"),
        factor_frame.join(common_dates, on="date", how="inner").select(["date", *symbols]).sort("date"),
    )


def _wide_matrix_pl(frame) -> pl.DataFrame:
    return _to_polars_matrix(frame).sort("date")


def _factor_coverage_pl(factors: pl.DataFrame) -> pl.DataFrame:
    symbols = [column for column in factors.columns if column != "date"]
    if not symbols:
        return pl.DataFrame({"date": factors["date"], "coverage": [0.0] * factors.height})
    return factors.select(
        [
            pl.col("date"),
            (pl.sum_horizontal([pl.col(symbol).is_not_null().cast(pl.Float64) for symbol in symbols]) / len(symbols)).alias("coverage"),
        ]
    )


def _weight_turnover_pl(weights: pl.DataFrame) -> pl.DataFrame:
    symbols = [column for column in weights.columns if column != "date"]
    if not symbols:
        return pl.DataFrame({"date": weights["date"], "turnover": [0.0] * weights.height})
    clean = weights.sort("date").with_columns([pl.col(symbol).fill_null(0.0).alias(symbol) for symbol in symbols])
    turnover_terms = [
        pl.col(symbol).diff().abs().fill_null(pl.col(symbol).abs())
        for symbol in symbols
    ]
    return clean.select(
        [
            pl.col("date"),
            (pl.sum_horizontal(turnover_terms) / 2.0).alias("turnover"),
        ]
    )




def top_bottom_weights(
    factors: pl.DataFrame,
    *,
    top_quantile: float = 0.2,
    bottom_quantile: float | None = None,
    long_weight: float = 1.0,
    short_weight: float = -1.0,
) -> pl.DataFrame:
    """Convert a date x symbol factor matrix into equal-weight long/short targets.

    On each date, symbols in the top factor quantile receive equal positive
    weights. If ``bottom_quantile`` is given, symbols in the bottom factor
    quantile receive equal negative weights. Ranking is cross-sectional, so the
    work is a window over ``date`` rather than a row loop.
    """
    if not 0 < top_quantile <= 1:
        raise ValueError("top_quantile must be in (0, 1]")
    if bottom_quantile is not None and not 0 < bottom_quantile <= 1:
        raise ValueError("bottom_quantile must be in (0, 1]")
    long, symbols = _factor_long_form(factors)
    if not symbols:
        return _wide_matrix_pl(factors).select("date")

    present = pl.col("value").is_not_null()
    count = pl.col("value").count().over("date")
    # int() truncation, matching max(1, int(n * quantile)).
    top_count = pl.max_horizontal(pl.lit(1, dtype=pl.Int64), (count * top_quantile).cast(pl.Int64))
    is_top = present & (pl.col("value").rank(method="ordinal", descending=True).over("date") <= top_count)
    long = long.with_columns(is_top.alias("_top"))

    if bottom_quantile is None or short_weight == 0:
        long = long.with_columns(pl.lit(False).alias("_bottom"))
    else:
        bottom_count = pl.max_horizontal(pl.lit(1, dtype=pl.Int64), (count * bottom_quantile).cast(pl.Int64))
        # Non-top names hold the lowest ascending ranks, so excluding the top
        # selection first falls out of the rank comparison.
        ascending = pl.col("value").rank(method="ordinal").over("date")
        long = long.with_columns((present & ~pl.col("_top") & (ascending <= bottom_count)).alias("_bottom"))

    top_size = pl.col("_top").sum().over("date")
    bottom_size = pl.col("_bottom").sum().over("date")
    weights = long.with_columns(
        pl.when(pl.col("_top"))
        .then(long_weight / top_size)
        .when(pl.col("_bottom"))
        .then(short_weight / bottom_size)
        .otherwise(0.0)
        .alias("value")
    ).drop("_top", "_bottom")
    return _factor_wide_form(weights, symbols)


def factor_coverage(factors: pl.DataFrame) -> pl.DataFrame:
    """Fraction of symbols with a non-null factor value on each date."""
    long, symbols = _factor_long_form(factors)
    if not symbols:
        return _empty_polars(["date", "coverage"])
    return (
        long.group_by("date")
        .agg((pl.col("value").is_not_null().sum() / pl.len()).alias("coverage"))
        .sort("date")
    )


def weight_turnover(weights) -> pl.DataFrame:
    """One-way turnover from target-weight changes."""
    long, symbols = _factor_long_form(weights)
    if not symbols:
        return _empty_polars(["date", "turnover"])
    filled = long.with_columns(pl.col("value").fill_null(0.0)).sort(["symbol", "date"])
    # The first date turns over from a flat book, so its change is the weight itself.
    changed = filled.with_columns(
        (pl.col("value") - pl.col("value").shift(1).over("symbol").fill_null(0.0)).abs().alias("delta")
    )
    return (
        changed.group_by("date").agg((pl.col("delta").sum() / 2.0).alias("turnover")).sort("date")
    )


def _factor_long_form(factors: pl.DataFrame) -> tuple[pl.DataFrame, list[str]]:
    long, symbols = _to_long(factors)
    return long, sorted(symbols)
