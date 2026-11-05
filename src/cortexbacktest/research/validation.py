"""Time-series validation and multiple-testing controls for research sweeps."""

from __future__ import annotations

from dataclasses import dataclass
import math
from statistics import NormalDist
from typing import Any, Callable

import numpy as np


@dataclass(frozen=True, slots=True)
class WalkForwardSplit:
    """Expanding or rolling train/test windows over ordered observations."""

    train_size: int
    test_size: int
    step_size: int | None = None
    embargo: int = 0
    expanding: bool = True

    def __post_init__(self) -> None:
        counts = (self.train_size, self.test_size, self.embargo,
                  self.test_size if self.step_size is None else self.step_size)
        if any(isinstance(n, bool) or not isinstance(n, (int, np.integer)) for n in counts):
            raise ValueError("train_size/test_size/step_size/embargo must be integers")
        if self.train_size < 1 or self.test_size < 1 or self.embargo < 0:
            raise ValueError("train_size/test_size must be positive and embargo non-negative")
        if self.step_size is not None and self.step_size < 1:
            raise ValueError("step_size must be positive")

    def split(self, observations) -> list[tuple[np.ndarray, np.ndarray]]:
        count = len(observations)
        step = self.step_size or self.test_size
        splits = []
        train_end = self.train_size
        while train_end + self.embargo + self.test_size <= count:
            train_start = 0 if self.expanding else train_end - self.train_size
            test_start = train_end + self.embargo
            splits.append((np.arange(train_start, train_end), np.arange(test_start, test_start + self.test_size)))
            train_end += step
        return splits


@dataclass(frozen=True, slots=True)
class WalkForwardResult:
    """Materialized output from one walk-forward evaluator call."""

    split: int
    train_indices: np.ndarray
    test_indices: np.ndarray
    value: Any


def run_walk_forward(
    observations,
    evaluator: Callable[[np.ndarray, np.ndarray], Any],
    *,
    splitter: WalkForwardSplit,
) -> list[WalkForwardResult]:
    """Evaluate a research or backtest callback independently per time split."""
    return [
        WalkForwardResult(
            split=split_number,
            train_indices=train.copy(),
            test_indices=test.copy(),
            value=evaluator(train.copy(), test.copy()),
        )
        for split_number, (train, test) in enumerate(splitter.split(observations))
    ]


def purged_walk_forward_splits(
    observations,
    *,
    train_size: int,
    test_size: int,
    label_horizon: int,
    step_size: int | None = None,
    embargo: int = 0,
    expanding: bool = True,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create walk-forward splits after removing training labels overlapping test.

    A training label at ``t`` is excluded when its forward horizon reaches the
    test window; embargo excludes additional observations immediately before it.
    """
    if isinstance(label_horizon, bool) or not isinstance(label_horizon, (int, np.integer)) or label_horizon < 0:
        raise ValueError("label_horizon must be a non-negative integer")
    base = WalkForwardSplit(train_size, test_size, step_size, embargo, expanding)
    result = []
    for train, test in base.split(observations):
        first_test = int(test[0])
        keep = train + label_horizon < first_test - embargo
        result.append((train[keep], test))
    return result


def deflated_sharpe_ratio(
    sharpe: float,
    *,
    observations: int,
    trials: int = 1,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Probability that Sharpe exceeds the expected best result from trials.

    Uses the Bailey–López de Prado approximation and returns a value in [0, 1].
    ``trials`` must be the effective number of configurations considered, not
    only the configuration selected after inspection.
    """
    if observations < 2 or trials < 1 or kurtosis <= 0:
        raise ValueError("observations >= 2, trials >= 1, and positive kurtosis are required")
    variance = (1.0 + 0.5 * sharpe * sharpe) / (observations - 1)
    sigma = variance**0.5
    normal = NormalDist()
    expected_max = 0.0
    if trials > 1:
        euler_gamma = 0.5772156649015329
        expected_max = sigma * (
            (1.0 - euler_gamma) * normal.inv_cdf(1.0 - 1.0 / trials)
            + euler_gamma * normal.inv_cdf(1.0 - 1.0 / (trials * np.e))
        )
    denominator = (1.0 - skewness * sharpe + ((kurtosis - 1.0) / 4.0) * sharpe * sharpe) ** 0.5
    if denominator == 0:
        return 0.0
    return float(normal.cdf((sharpe - expected_max) * (observations - 1) ** 0.5 / denominator))


@dataclass(frozen=True, slots=True)
class TTestResult:
    """Two-sided test of H0: mean == 0."""

    mean: float
    t_stat: float
    p_value: float
    observations: int


@dataclass(frozen=True, slots=True)
class MultipleTestingResult:
    """Benjamini-Hochberg adjustment over a family of p-values."""

    p_values: np.ndarray
    q_values: np.ndarray
    rejected: np.ndarray
    alpha: float

    @property
    def discoveries(self) -> int:
        return int(self.rejected.sum())


def _beta_continued_fraction(a: float, b: float, x: float) -> float:
    """Modified Lentz evaluation of the beta continued fraction."""
    max_iterations = 300
    epsilon = 3.0e-16
    tiny = 1.0e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, max_iterations + 1):
        m2 = 2 * m
        numerator = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + numerator * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + numerator / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        numerator = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + numerator * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + numerator / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < epsilon:
            break
    return h


def regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    """``I_x(a, b)``, implemented here so p-values do not require SciPy."""
    if not 0.0 <= x <= 1.0:
        raise ValueError("x must be in [0, 1]")
    if x == 0.0:
        return 0.0
    if x == 1.0:
        return 1.0
    log_prefix = (
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    prefix = math.exp(log_prefix)
    if x < (a + 1.0) / (a + b + 2.0):
        return prefix * _beta_continued_fraction(a, b, x) / a
    return 1.0 - prefix * _beta_continued_fraction(b, a, 1.0 - x) / b


def student_t_two_sided_p_value(t_stat: float, degrees_of_freedom: int) -> float:
    """Two-sided Student's t p-value.

    SciPy is not a runtime dependency of this package, so the tail is evaluated
    from the regularized incomplete beta directly.
    """
    if degrees_of_freedom < 1:
        raise ValueError("degrees_of_freedom must be positive")
    if not math.isfinite(t_stat):
        return float("nan")
    df = float(degrees_of_freedom)
    return regularized_incomplete_beta(df / 2.0, 0.5, df / (df + t_stat * t_stat))


def t_test_mean_zero(values) -> TTestResult:
    """Test whether a series (for example IC by date) has a non-zero mean."""
    sample = np.asarray(values, dtype=float)
    sample = sample[np.isfinite(sample)]
    observations = int(sample.size)
    if observations < 2:
        return TTestResult(
            mean=float(sample.mean()) if observations else float("nan"),
            t_stat=float("nan"),
            p_value=float("nan"),
            observations=observations,
        )
    mean = float(sample.mean())
    standard_deviation = float(sample.std(ddof=1))
    if standard_deviation == 0.0:
        # Zero variance is the limiting case: a constant non-zero series drives
        # t to +/-inf and p to 0, while a constant zero series carries no signal
        # and leaves both undefined.
        if mean == 0.0:
            return TTestResult(mean=mean, t_stat=float("nan"), p_value=float("nan"), observations=observations)
        return TTestResult(
            mean=mean,
            t_stat=math.copysign(float("inf"), mean),
            p_value=0.0,
            observations=observations,
        )
    t_stat = mean / (standard_deviation / math.sqrt(observations))
    return TTestResult(
        mean=mean,
        t_stat=float(t_stat),
        p_value=float(student_t_two_sided_p_value(t_stat, observations - 1)),
        observations=observations,
    )


def benjamini_hochberg(p_values, *, alpha: float = 0.05) -> MultipleTestingResult:
    """Control the false discovery rate across a family of tests.

    A parameter sweep tests many configurations at once, so the smallest p-value
    among them is optimistic. This returns BH-adjusted q-values and the
    rejection flags at ``alpha``. Complements ``deflated_sharpe_ratio``, which
    penalises Sharpe selection rather than per-test significance.

    NaN p-values (too few observations to test) are carried through as NaN and
    never rejected, and are excluded from the family size.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    p = np.asarray(p_values, dtype=float)
    if p.ndim != 1:
        raise ValueError("p_values must be one dimensional")
    q_values = np.full(p.shape, np.nan)
    rejected = np.zeros(p.shape, dtype=bool)
    testable = np.isfinite(p)
    count = int(testable.sum())
    if count == 0:
        return MultipleTestingResult(p_values=p, q_values=q_values, rejected=rejected, alpha=alpha)
    if np.any((p[testable] < 0.0) | (p[testable] > 1.0)):
        raise ValueError("p_values must be in [0, 1]")
    positions = np.flatnonzero(testable)
    ordered_positions = positions[np.argsort(p[positions], kind="stable")]
    ranks = np.arange(1, count + 1, dtype=float)
    raw = p[ordered_positions] * count / ranks
    # Enforce monotone non-decreasing q-values from the largest p downwards.
    stepped = np.minimum.accumulate(raw[::-1])[::-1]
    stepped = np.clip(stepped, 0.0, 1.0)
    q_values[ordered_positions] = stepped
    rejected[ordered_positions] = stepped <= alpha
    return MultipleTestingResult(p_values=p, q_values=q_values, rejected=rejected, alpha=alpha)


@dataclass(frozen=True, slots=True)
class DataSnoopingResult:
    """Whether the best of many searched strategies really beats its benchmark.

    ``p_value`` is the headline: the probability of seeing a maximum this large
    when no strategy in the family beats the benchmark. ``best`` indexes the
    winning column so a caller can report which configuration was tested, not
    only that something won. Hansen's test also reports the bounding
    ``p_lower``/``p_upper``; White's reality check leaves them None.
    """

    statistic: float
    p_value: float
    best: int
    best_mean: float
    resamples: int
    block_size: float
    p_lower: float | None = None
    p_upper: float | None = None


def stationary_bootstrap_indices(
    observations: int,
    *,
    resamples: int,
    block_size: float,
    seed: int | None = None,
) -> np.ndarray:
    """Politis-Romano stationary bootstrap index matrix, shape (resamples, observations).

    Blocks of geometric length (mean ``block_size``) are drawn and wrapped around
    the sample, which preserves the serial dependence an ordinary iid bootstrap
    destroys — option strategy returns are autocorrelated, so an iid resample
    would understate the variance of the maximum and make every strategy look
    significant. ``block_size = 1`` degenerates to the iid bootstrap.
    """
    if observations < 2 or resamples < 1:
        raise ValueError("observations >= 2 and resamples >= 1 are required")
    if not math.isfinite(block_size) or block_size < 1.0:
        raise ValueError("block_size must be finite and at least 1")
    generator = np.random.default_rng(seed)
    restart = 1.0 / block_size
    indices = np.empty((resamples, observations), dtype=np.int64)
    indices[:, 0] = generator.integers(0, observations, resamples)
    starts = generator.integers(0, observations, (resamples, observations - 1))
    fresh = generator.random((resamples, observations - 1)) < restart
    for step in range(1, observations):
        continued = (indices[:, step - 1] + 1) % observations
        indices[:, step] = np.where(fresh[:, step - 1], starts[:, step - 1], continued)
    return indices


def _snooping_inputs(performance, resamples: int, block_size: float) -> np.ndarray:
    matrix = np.asarray(performance, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[1] < 1:
        raise ValueError("performance must be a T x M matrix with T >= 2 and M >= 1")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("performance must be finite")
    if resamples < 1:
        raise ValueError("resamples must be positive")
    if not math.isfinite(block_size) or block_size < 1.0:
        raise ValueError("block_size must be finite and at least 1")
    return matrix


def reality_check(
    performance,
    *,
    resamples: int = 1000,
    block_size: float = 5.0,
    seed: int | None = None,
) -> DataSnoopingResult:
    """White (2000) reality check over a family of searched strategies.

    ``performance`` is a T x M matrix of per-period performance *differences*
    against the benchmark — strategy return minus benchmark return, or minus
    zero for an absolute-return question. Every strategy that was tried belongs
    in it, including the ones abandoned: the null being tested is "no strategy
    in this family beats the benchmark", and leaving the failures out is exactly
    the data snooping the test exists to price in.

    The null distribution comes from the stationary bootstrap, recentred on the
    observed means, so it answers "how large a maximum does a search of this
    size produce by luck?" rather than "is this one strategy significant?".

    Its known weakness is why `superior_predictive_ability` exists: padding the
    family with strategies that are obviously terrible drags the bootstrap
    maximum around and costs power.
    """
    matrix = _snooping_inputs(performance, resamples, block_size)
    periods = matrix.shape[0]
    means = matrix.mean(axis=0)
    best = int(np.argmax(means))
    statistic = float(math.sqrt(periods) * means[best])
    indices = stationary_bootstrap_indices(
        periods, resamples=resamples, block_size=block_size, seed=seed
    )
    resampled = matrix[indices].mean(axis=1)
    draws = np.sqrt(periods) * (resampled - means)
    p_value = float(np.mean(draws.max(axis=1) > statistic))
    return DataSnoopingResult(
        statistic=statistic,
        p_value=p_value,
        best=best,
        best_mean=float(means[best]),
        resamples=resamples,
        block_size=block_size,
    )


def superior_predictive_ability(
    performance,
    *,
    resamples: int = 1000,
    block_size: float = 5.0,
    seed: int | None = None,
) -> DataSnoopingResult:
    """Hansen (2005) SPA test: the reality check, studentised and recentred.

    Same input convention as `reality_check`. Two changes make it the better
    default when the family contains many weak candidates, which a machine
    learning sweep always does:

    * each column is divided by its own bootstrap standard error, so a noisy
      strategy cannot dominate the maximum simply by being noisy;
    * columns whose sample mean is far enough below zero are recentred to zero
      rather than to their own mean, which stops hopeless candidates from
      inflating the null distribution.

    ``p_value`` is Hansen's consistent p-value; ``p_lower``/``p_upper`` bound it
    by recentring nothing and everything respectively. A wide gap between them
    means the verdict rests on the recentring threshold rather than the data.
    """
    matrix = _snooping_inputs(performance, resamples, block_size)
    periods, count = matrix.shape
    means = matrix.mean(axis=0)
    indices = stationary_bootstrap_indices(
        periods, resamples=resamples, block_size=block_size, seed=seed
    )
    resampled = matrix[indices].mean(axis=1)
    centred = np.sqrt(periods) * (resampled - means)
    # Hansen's omega: the bootstrap standard error of the recentred mean. A
    # constant column has none, and studentising it would divide by zero, so it
    # is held out of the maximum rather than silently scaled to infinity.
    omega = centred.std(axis=0, ddof=1)
    usable = omega > 0
    if not np.any(usable):
        raise ValueError("every strategy is constant; SPA is undefined")
    scaled = np.where(usable, np.sqrt(periods) * means / np.where(usable, omega, 1.0), -np.inf)
    best = int(np.argmax(scaled))
    statistic = float(max(0.0, scaled[best]))
    threshold = omega * math.sqrt(2.0 * math.log(math.log(periods))) / math.sqrt(periods) \
        if periods > math.e else np.zeros(count)
    variants = {
        "p_lower": np.where(means >= 0.0, means, 0.0),
        "p_value": np.where(means >= -threshold, means, 0.0),
        "p_upper": means,
    }
    p_values: dict[str, float] = {}
    for name, recentred in variants.items():
        shifted = np.sqrt(periods) * (resampled - recentred)
        studentised = np.where(usable, shifted / np.where(usable, omega, 1.0), -np.inf)
        maxima = np.maximum(studentised.max(axis=1), 0.0)
        p_values[name] = float(np.mean(maxima > statistic))
    return DataSnoopingResult(
        statistic=statistic,
        p_value=p_values["p_value"],
        best=best,
        best_mean=float(means[best]),
        resamples=resamples,
        block_size=block_size,
        p_lower=p_values["p_lower"],
        p_upper=p_values["p_upper"],
    )


@dataclass(frozen=True, slots=True)
class HaircutResult:
    """Harvey-Liu haircut of a reported Sharpe ratio for multiple testing."""

    sharpe: float
    haircut_sharpe: float
    t_stat: float
    adjusted_t_stat: float
    p_value: float
    adjusted_p_value: float
    trials: int
    method: str

    @property
    def haircut(self) -> float:
        """Fraction of the reported Sharpe that multiple testing takes away."""
        return 1.0 - self.haircut_sharpe / self.sharpe


def _t_stat_for_two_sided_p(p_value: float, degrees_of_freedom: int) -> float:
    """Invert `student_t_two_sided_p_value` by bisection; the same tail both ways."""
    if p_value >= 1.0:
        return 0.0
    low, high = 0.0, 1.0
    while student_t_two_sided_p_value(high, degrees_of_freedom) > p_value:
        high *= 2.0
        if high > 1e12:
            return high
    for _ in range(200):
        middle = (low + high) / 2.0
        if student_t_two_sided_p_value(middle, degrees_of_freedom) > p_value:
            low = middle
        else:
            high = middle
    return (low + high) / 2.0


def sharpe_haircut(
    sharpe: float,
    *,
    observations: int,
    trials: int = 1,
    method: str = "bonferroni",
) -> HaircutResult:
    """Harvey-Liu (2015): what a reported Sharpe is worth after multiple testing.

    ``sharpe`` is the *per-period* sample Sharpe over ``observations`` periods —
    divide an annualised figure by sqrt(periods per year) before calling — and
    ``trials`` is how many strategies were searched to find it, including the
    ones nobody wrote down.

    The reported Sharpe is turned into a t-statistic, its two-sided p-value is
    adjusted for the size of the search, and the adjusted p-value is turned back
    into a Sharpe. `haircut` is the fraction lost.

    ``method`` selects the adjustment. For the most significant of ``trials``
    tests, Holm's step-down coincides with Bonferroni, so both are exposed under
    the names researchers expect; ``bhy`` (Benjamini-Hochberg-Yekutieli)
    adds a harmonic penalty here. This treats the strategy as the best of the search,
    which is the case the haircut is for.
    """
    if method not in ("bonferroni", "holm", "bhy"):
        raise ValueError("method must be bonferroni, holm or bhy")
    if not math.isfinite(sharpe) or sharpe <= 0:
        raise ValueError("haircut applies to a positive reported Sharpe")
    if observations < 2 or trials < 1:
        raise ValueError("observations >= 2 and trials >= 1 are required")
    t_stat = sharpe * math.sqrt(observations)
    p_value = student_t_two_sided_p_value(t_stat, observations - 1)
    multiplier = float(trials)
    if method == "bhy":
        multiplier *= float(sum(1.0 / rank for rank in range(1, trials + 1)))
    adjusted_p = min(1.0, p_value * multiplier)
    adjusted_t = _t_stat_for_two_sided_p(adjusted_p, observations - 1)
    return HaircutResult(
        sharpe=float(sharpe),
        haircut_sharpe=float(sharpe * adjusted_t / t_stat) if t_stat > 0 else 0.0,
        t_stat=float(t_stat),
        adjusted_t_stat=float(adjusted_t),
        p_value=float(p_value),
        adjusted_p_value=float(adjusted_p),
        trials=int(trials),
        method=method,
    )
