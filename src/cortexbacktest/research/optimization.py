"""Lightweight portfolio construction utilities."""

from __future__ import annotations

import polars as pl

from .frames import _to_polars_matrix
from .enhanced_index import optimize_enhanced_index  # noqa: F401


def select_top_n(scores: pl.DataFrame, *, n: int, ascending: bool = False) -> pl.DataFrame:
    """Select top `n` symbols per date from a wide Polars score matrix."""
    if n < 1:
        raise ValueError("n must be >= 1")
    data = _wide_matrix_pl(scores)
    symbols = _symbols(data)
    if not symbols:
        return data
    rank_descending = not ascending
    long_form = data.unpivot(index="date", variable_name="symbol", value_name="score").drop_nulls("score")
    selected = (
        long_form.with_columns(pl.col("score").rank(method="ordinal", descending=rank_descending).over("date").alias("rank"))
        .with_columns((pl.col("rank") <= n).alias("selected"))
        .pivot(index="date", on="symbol", values="selected")
    )
    selected = selected.with_columns(pl.lit(False).alias(symbol) for symbol in symbols
                                     if symbol not in selected.columns)
    return (
        data.select("date")
        .join(selected, on="date", how="left")
        .select([pl.col("date"), *[pl.col(symbol).fill_null(False).cast(pl.Boolean).alias(symbol) for symbol in symbols]])
    )


def weights_from_signals(signals: pl.DataFrame, *, gross_weight: float = 1.0) -> pl.DataFrame:
    """Convert wide boolean selections to row-wise equal target weights."""
    if gross_weight < 0:
        raise ValueError("gross_weight must be non-negative")
    data = _wide_matrix_pl(signals)
    symbols = _symbols(data)
    if not symbols:
        return data
    selected = [pl.col(symbol).fill_null(False).cast(pl.Boolean) for symbol in symbols]
    return _equal_weights(data, symbols, selected, gross_weight)


def equal_weight(scores: pl.DataFrame, *, gross_weight: float = 1.0, long_only: bool = True) -> pl.DataFrame:
    """Create row-wise equal weights over available symbols as a Polars wide matrix."""
    if gross_weight < 0:
        raise ValueError("gross_weight must be non-negative")
    data = _wide_matrix_pl(scores)
    symbols = _symbols(data)
    if not symbols:
        return data
    selected = [
        (pl.col(symbol).is_not_null() & (pl.col(symbol) > 0 if long_only else pl.lit(True)))
        for symbol in symbols
    ]
    return _equal_weights(data, symbols, selected, gross_weight)


def _equal_weights(data: pl.DataFrame, symbols: list[str], selected: list[pl.Expr], gross_weight: float) -> pl.DataFrame:
    count = pl.sum_horizontal([pl.when(expr).then(1.0).otherwise(0.0) for expr in selected])
    expressions = [
        pl.when((count > 0) & expr)
        .then(gross_weight / count)
        .otherwise(0.0)
        .alias(symbol)
        for symbol, expr in zip(symbols, selected, strict=True)
    ]
    return data.select([pl.col("date"), *expressions])


def normalize_weights(weights: pl.DataFrame, *, gross_weight: float = 1.0) -> pl.DataFrame:
    """Normalize each row so absolute exposure equals `gross_weight`."""
    if gross_weight < 0:
        raise ValueError("gross_weight must be non-negative")
    data = _wide_matrix_pl(weights)
    symbols = _symbols(data)
    if not symbols:
        return data
    gross = pl.sum_horizontal([pl.col(symbol).fill_null(0.0).abs() for symbol in symbols])
    expressions = [
        pl.when(gross == 0)
        .then(pl.col(symbol).fill_null(0.0))
        .otherwise(pl.col(symbol).fill_null(0.0) / gross * gross_weight)
        .alias(symbol)
        for symbol in symbols
    ]
    return data.select([pl.col("date"), *expressions])


def inverse_volatility_weights(
    returns: pl.DataFrame,
    *,
    window: int = 20,
    min_periods: int | None = None,
    gross_weight: float = 1.0,
) -> pl.DataFrame:
    """Build date x symbol inverse-volatility weights as a Polars wide matrix."""
    if window < 1:
        raise ValueError("window must be >= 1")
    if min_periods is None:
        min_periods = window
    data = _wide_matrix_pl(returns)
    volatility = pl.col(_symbols(data)).rolling_std(window_size=window, min_samples=min_periods, ddof=1)
    inverse = data.select("date", (1.0 / pl.when(volatility == 0).then(None).otherwise(volatility)).name.keep())
    return normalize_weights(inverse, gross_weight=gross_weight)


def _wide_matrix_pl(frame) -> pl.DataFrame:
    data = _to_polars_matrix(frame).sort("date")
    return data.select(["date", *sorted(_symbols(data))])


def _symbols(frame: pl.DataFrame) -> list[str]:
    return [column for column in frame.columns if column != "date"]


def cap_weights(
    weights: pl.DataFrame,
    *,
    max_weight: float,
    gross_weight: float | None = None,
    iterations: int = 10,
) -> pl.DataFrame:
    """Cap absolute position weights per date and redistribute the remainder.

    Each pass caps the offenders and spreads what is left across the rest, so
    repeated passes converge. The loop is over passes, not rows: one pass
    updates every date at once.
    """
    if max_weight <= 0:
        raise ValueError("max_weight must be positive")
    if iterations < 1:
        raise ValueError("iterations must be >= 1")
    if gross_weight is not None and gross_weight < 0:
        raise ValueError("gross_weight must be non-negative")
    data = _wide_matrix_pl(weights)
    symbols = _symbols(data)
    if not symbols:
        return data

    filled = data.select([pl.col("date"), *[pl.col(symbol).fill_null(0.0).alias(symbol) for symbol in symbols]])
    target = (
        pl.sum_horizontal([pl.col(symbol).abs() for symbol in symbols])
        if gross_weight is None
        else pl.lit(float(gross_weight))
    )
    result = filled.with_columns(target.alias("_target"))
    gross = pl.sum_horizontal([pl.col(symbol).abs() for symbol in symbols])
    result = result.select(
        [
            pl.col("date"),
            pl.col("_target"),
            *[
                pl.when(gross == 0).then(pl.col(symbol)).otherwise(pl.col(symbol) / gross * pl.col("_target")).alias(symbol)
                for symbol in symbols
            ],
        ]
    )

    flags = [f"_over_{index}" for index in range(len(symbols))]
    for _ in range(iterations):
        # Materialize the over-limit mask before capping. Re-deriving it from the
        # capped frame would flip to False (|w| == max_weight is not > it) and
        # redistribute the very positions that were just pinned.
        result = result.with_columns(
            [(pl.col(symbol).abs() > max_weight).alias(flag) for symbol, flag in zip(symbols, flags, strict=True)]
        )
        if not result.select(pl.any_horizontal([pl.col(flag) for flag in flags]).any()).item():
            result = result.drop(flags)
            break
        capped_gross = pl.sum_horizontal(
            [pl.when(pl.col(flag)).then(pl.lit(max_weight)).otherwise(0.0) for flag in flags]
        )
        free_gross = pl.sum_horizontal(
            [pl.when(pl.col(flag)).then(0.0).otherwise(pl.col(symbol).abs()) for symbol, flag in zip(symbols, flags, strict=True)]
        )
        remaining = pl.max_horizontal(pl.col("_target") - capped_gross, pl.lit(0.0))
        result = result.with_columns(
            [
                pl.when(pl.col(flag))
                .then(pl.lit(max_weight) * pl.col(symbol).sign())
                .when((remaining == 0) | (free_gross == 0))
                .then(pl.col(symbol))
                .otherwise(pl.col(symbol) / free_gross * remaining)
                .alias(symbol)
                for symbol, flag in zip(symbols, flags, strict=True)
            ]
        ).drop(flags)
    return result.select(["date", *symbols])


def target_volatility_scale(
    returns: pl.DataFrame,
    *,
    target_volatility: float,
    realized_volatility: float | None = None,
    periods_per_year: int = 252,
    max_leverage: float | None = None,
) -> float:
    """Compute the leverage scale needed to hit an annualized volatility target."""
    if target_volatility < 0:
        raise ValueError("target_volatility must be non-negative")
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")
    if max_leverage is not None and max_leverage <= 0:
        raise ValueError("max_leverage must be positive")
    volatility = realized_volatility
    if volatility is None:
        clean = _return_series(returns).drop_nulls()
        volatility = float(clean.std(ddof=1) * periods_per_year**0.5) if len(clean) > 1 else 0.0
    if volatility is None or volatility <= 0:
        return 0.0
    scale = target_volatility / volatility
    if max_leverage is not None:
        scale = min(scale, max_leverage)
    return float(scale)


def apply_volatility_target(
    weights: pl.DataFrame,
    portfolio_returns,
    *,
    target_volatility: float,
    window: int = 20,
    periods_per_year: int = 252,
    max_leverage: float | None = None,
) -> pl.DataFrame:
    """Scale target weights by trailing realized portfolio volatility."""
    if window < 2:
        raise ValueError("window must be >= 2")
    data = _wide_matrix_pl(weights)
    symbols = _symbols(data)
    if not symbols:
        return data
    if isinstance(portfolio_returns, pl.DataFrame) and "date" in portfolio_returns.columns:
        portfolio_returns = _to_polars_matrix(portfolio_returns).sort("date")
        if not data["date"].equals(portfolio_returns["date"]):
            raise ValueError("portfolio_returns dates must match the weight matrix exactly")
    series = _return_series(portfolio_returns)
    if len(series) != data.height:
        raise ValueError("portfolio_returns must align with the weight matrix rows")

    trailing = series.to_frame("return").select(
        (pl.col("return").rolling_std(window_size=window, ddof=1) * periods_per_year**0.5).alias("volatility")
    )["volatility"]
    scales = [
        target_volatility_scale(
            series,
            target_volatility=target_volatility,
            realized_volatility=float(volatility),
            periods_per_year=periods_per_year,
            max_leverage=max_leverage,
        )
        # A window that has not filled yet has no volatility estimate, so there
        # is nothing to scale against and the row stays flat.
        if volatility is not None
        else 0.0
        for volatility in trailing
    ]
    scale_column = pl.Series("_scale", scales, dtype=pl.Float64)
    return data.with_columns(scale_column).select(
        [pl.col("date"), *[(pl.col(symbol).fill_null(0.0) * pl.col("_scale")).alias(symbol) for symbol in symbols]]
    )


def _return_series(returns: pl.DataFrame) -> pl.Series:
    if isinstance(returns, pl.Series):
        return returns
    if isinstance(returns, pl.DataFrame):
        columns = [column for column in returns.columns if column != "date"]
        if len(columns) != 1:
            raise ValueError("returns frame must hold exactly one value column beside date")
        return returns[columns[0]]
    return pl.Series("return", list(returns), dtype=pl.Float64)


def _allocation_data(returns: pl.DataFrame):
    import numpy as np

    data = _wide_matrix_pl(returns)
    symbols = _symbols(data)
    if (not symbols or len(symbols) > 200 or data.height < 2
            or data["date"].null_count() or data["date"].n_unique() != data.height):
        raise ValueError("allocation requires 2+ unique dated rows and 1..200 instruments")
    if any(not data.schema[s].is_numeric() for s in symbols):
        raise ValueError("allocation returns must be numeric")
    values = data.select(symbols).to_numpy().astype(float)
    if not np.isfinite(values).all():
        raise ValueError("allocation returns must be complete and finite")
    cov = np.atleast_2d(np.cov(values, rowvar=False, ddof=1))
    if np.any(np.diag(cov) <= 0) or not np.isfinite(cov).all():
        raise ValueError("allocation requires positive finite variances")
    return symbols, cov


def hierarchical_risk_parity(returns: pl.DataFrame) -> dict[str, float]:
    "HRP with deterministic single linkage and recursive bisection; trailing data only."
    import numpy as np

    symbols, cov = _allocation_data(returns)
    n = len(symbols)
    std = np.sqrt(np.diag(cov))
    distance = np.sqrt(np.maximum(0, (1 - np.clip(cov / np.outer(std, std), -1, 1)) / 2))
    clusters = [[i] for i in range(n)]
    # Lexical symbol order breaks equal-distance ties reproducibly.
    while len(clusters) > 1:
        _, a, b = min((min(distance[i, j] for i in left for j in right), a, b)
                     for a, left in enumerate(clusters) for b, right in enumerate(clusters)
                     if a < b)
        merged = clusters[a] + clusters[b]
        clusters = [cluster for k, cluster in enumerate(clusters) if k not in (a, b)]
        clusters.append(merged)
    weights = np.ones(n)
    pending = clusters
    def variance(indices):
        sub = cov[np.ix_(indices, indices)]
        inverse = 1 / np.diag(sub)
        w = inverse / inverse.sum()
        return float(w @ sub @ w)
    while pending:
        cluster = pending.pop()
        if len(cluster) < 2:
            continue
        middle = len(cluster) // 2
        left, right = cluster[:middle], cluster[middle:]
        v_left, v_right = variance(left), variance(right)
        if v_left + v_right <= 0:
            raise ValueError("degenerate HRP cluster variance")
        fraction = v_right / (v_left + v_right)
        weights[left] *= fraction
        weights[right] *= 1 - fraction
        pending.extend([left, right])
    weights /= weights.sum()
    if not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("invalid HRP solution")
    return dict(zip(symbols, map(float, weights), strict=True))


def optimize_weights(
    returns: pl.DataFrame, *, previous: dict[str, float] | None = None,
    expected_returns: dict[str, float] | None = None, risk_aversion: float = 1.0,
    max_weight: float = 1.0, max_turnover: float | None = None,
) -> dict[str, float]:
    "Solve long-only GMV/MVO with explicit L1 asset-weight turnover (cash excluded)."
    import numpy as np

    symbols, cov = _allocation_data(returns)
    if (type(max_weight) not in (int, float) or not np.isfinite(max_weight)
            or not 0 < max_weight <= 1 or len(symbols) * max_weight < 1 - 1e-10):
        raise ValueError("infeasible or invalid max_weight")
    if (type(risk_aversion) not in (int, float) or not np.isfinite(risk_aversion)
            or risk_aversion <= 0):
        raise ValueError("risk_aversion must be finite and positive")
    def vector(mapping, label):
        if set(mapping) != set(symbols):
            raise ValueError(f"{label} instrument labels must match returns exactly")
        if any(type(mapping[s]) not in (int, float) for s in symbols):
            raise ValueError(f"{label} weights/returns must be numeric")
        values = np.array([mapping[s] for s in symbols], dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"{label} must be finite")
        return values
    prior = None if previous is None else vector(previous, "previous")
    if prior is not None and (np.any(prior < 0) or prior.sum() > 1 + 1e-8):
        raise ValueError("previous must be long-only and have total weight <= 1")
    mu = np.zeros(len(symbols)) if expected_returns is None else vector(expected_returns, "expected_returns")
    if max_turnover is not None:
        if (type(max_turnover) not in (int, float) or not np.isfinite(max_turnover)
                or max_turnover < 0 or prior is None):
            raise ValueError("turnover requires a finite nonnegative budget and previous weights")
    try:
        import cvxpy as cp
        from cvxpy.error import SolverError
    except ModuleNotFoundError as error:
        if error.name not in {"cvxpy", "cvxpy.error"}:
            raise
        raise ImportError("constrained allocation requires the optional optimizer extra") from error
    weights = cp.Variable(len(symbols))
    constraints = [weights >= 0, weights <= max_weight, cp.sum(weights) == 1]
    if max_turnover is not None:
        constraints.append(cp.norm1(weights - prior) <= max_turnover)
    # Normalize the entire objective, preserving its minimizer for small daily covariances.
    scale = max(float(np.max(np.diag(cov))) * risk_aversion, float(np.max(np.abs(mu))), 1e-12)
    problem = cp.Problem(cp.Minimize((risk_aversion * cp.quad_form(weights, cp.psd_wrap(cov))
                                     - mu @ weights) / scale), constraints)
    try:
        problem.solve(solver="CLARABEL")
    except SolverError as error:
        raise ValueError("allocation solver failed") from error
    if problem.status != cp.OPTIMAL or weights.value is None:
        raise ValueError(f"allocation solver rejected status: {problem.status}")
    result = np.asarray(weights.value).ravel()
    tolerance = 1e-7
    if (not np.isfinite(result).all() or abs(result.sum() - 1) > tolerance
            or np.any(result < -tolerance) or np.any(result > max_weight + tolerance)
            or (max_turnover is not None and np.abs(result - prior).sum() > max_turnover + tolerance)):
        raise ValueError("allocation solution violates constraints")
    # Remove solver noise only after validating; recheck turnover and upper bound afterwards.
    result = np.maximum(result, 0)
    result /= result.sum()
    if (np.any(result > max_weight + tolerance)
            or (max_turnover is not None and np.abs(result - prior).sum() > max_turnover + tolerance)):
        raise ValueError("allocation normalization violates constraints")
    return dict(zip(symbols, map(float, result), strict=True))
