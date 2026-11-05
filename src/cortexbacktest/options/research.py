"""Paper-informed research primitives; not empirical paper replications."""
from dataclasses import dataclass
from datetime import datetime, timedelta
from itertools import combinations
from math import isfinite, log, sqrt
from statistics import NormalDist
from typing import Iterable, Mapping

import numpy as np

from .backtest import OptionAccountView
from .contracts import OptionContract, OptionMarketFrame, QuotePolicy, utc
from .pricing import option_right


def historical_volatility(prices: Iterable[float], *, periods_per_year: int = 252) -> float:
    """Sample SD of simple daily returns, annualized; caller supplies PIT window.

    Goyal-Saretto uses the preceding 12 calendar months, not a fixed 20-day window.
    Prices must consistently reflect corporate actions and the declared return convention.
    """
    values = np.asarray(tuple(prices), dtype=float)
    if len(values) < 3 or not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("at least three finite positive historical prices required")
    if isinstance(periods_per_year, bool) or periods_per_year <= 0:
        raise ValueError("positive annualization factor required")
    return float(np.std(values[1:] / values[:-1] - 1, ddof=1) * sqrt(periods_per_year))


def goyal_saretto_signal(historical_vol: float, call_iv: float, put_iv: float) -> float:
    """log(HV) - log(mean(ATM call IV, ATM put IV)); matched expiry required."""
    if not all(isfinite(x) and x > 0 for x in (historical_vol, call_iv, put_iv)):
        raise ValueError("HV and observed call/put IV must be finite and positive")
    return log(historical_vol) - log((call_iv + put_iv) / 2)


def select_option(contracts: Iterable[OptionContract], frame: OptionMarketFrame, *,
                  underlying: str, right: str, target_dte: float = 30,
                  min_dte: float = 25, max_dte: float = 35,
                  target_delta: float | None = None,
                  policy: QuotePolicy | None = None) -> OptionContract | None:
    """DTE first, then ATM/delta distance, strike and symbol deterministic ties."""
    right = option_right(right)
    if not all(isfinite(x) for x in (target_dte, min_dte, max_dte)) or not (
            0 <= min_dte <= target_dte <= max_dte):
        raise ValueError("invalid DTE range")
    if target_delta is not None and (not isfinite(target_delta) or not -1 <= target_delta <= 1):
        raise ValueError("invalid target delta")
    policy = policy or QuotePolicy()
    spot_quote = frame.quotes.get(underlying)
    if spot_quote is None or policy.rejection(spot_quote, frame.timestamp, closing=True):
        return None
    candidates = []
    for contract in contracts:
        quote = frame.quotes.get(contract.symbol)
        dte = (contract.expiry - frame.timestamp).total_seconds() / 86400
        if (contract.underlying != underlying or contract.right != right
                or not min_dte <= dte <= max_dte or dte <= 0 or quote is None
                or policy.rejection(quote, frame.timestamp)):
            continue
        if target_delta is not None:
            if quote.delta is None:
                continue
            distance = abs(quote.delta - target_delta)
        else:
            distance = abs(contract.strike - spot_quote.mid)
        candidates.append((abs(dte - target_dte), distance, contract.strike, contract.symbol, contract))
    return min(candidates, key=lambda item: item[:4])[-1] if candidates else None


def delta_hedge_targets(contracts: Mapping[str, OptionContract], frame: OptionMarketFrame,
                        positions: Mapping[str, float], *, threshold: float = 0,
                        round_shares: bool = True) -> dict[str, float]:
    """Targets from currently held options and PIT supplied deltas, next-frame fills.

    Call only at the desired daily/weekly schedule. No model silently supplies IV.
    """
    if not isfinite(threshold) or threshold < 0:
        raise ValueError("threshold must be finite nonnegative")
    exposures = {c.underlying: 0.0 for c in contracts.values()}
    for symbol, quantity in positions.items():
        if symbol not in contracts or not quantity:
            continue
        contract = contracts[symbol]
        quote = frame.quotes.get(symbol)
        if quote is None or quote.delta is None:
            raise ValueError(f"missing point-in-time delta for {symbol}")
        exposures[contract.underlying] += quantity * contract.multiplier * quote.delta
    result = {}
    for symbol, delta in exposures.items():
        target = float(round(-delta)) if round_shares else -delta
        if abs(target - positions.get(symbol, 0)) > threshold:
            result[symbol] = target
    return result


@dataclass(frozen=True, slots=True)
class ScheduledOptionTargets:
    """Externally specified PIT option targets, optionally rebalance held delta.

    Schedule is a strategy parameter, NOT a future market-data feed. Targets at
    t execute after t; the first hedge follows actual option fills, with a lag.
    """
    contracts: tuple[OptionContract, ...]
    schedule: Mapping[datetime, Mapping[str, float]]
    hedge_every: timedelta | None = None
    hedge_threshold: float = 0

    def strategy(self):
        book = {c.symbol: c for c in self.contracts}
        schedule = {utc(t): dict(v) for t, v in self.schedule.items()}
        if self.hedge_every is not None and self.hedge_every <= timedelta(0):
            raise ValueError("positive hedge interval required")
        last_hedge = None
        def decide(frame: OptionMarketFrame, account: OptionAccountView):
            nonlocal last_hedge
            targets = dict(schedule.get(frame.timestamp, {}))
            if self.hedge_every is not None and (last_hedge is None or
                    frame.timestamp - last_hedge >= self.hedge_every):
                targets.update(delta_hedge_targets(book, frame, account.positions,
                                                   threshold=self.hedge_threshold))
                last_hedge = frame.timestamp
            return targets
        return decide


def purged_option_split(decision_times: Iterable[datetime], label_ends: Iterable[datetime], *,
                        test_start: datetime, test_end: datetime,
                        embargo: timedelta = timedelta(0)) -> tuple[np.ndarray, np.ndarray]:
    """Forward-only split of entire timestamp cross-sections, inclusive label ends.

    Require training labels strictly before test_start - embargo. Call separately
    for validation and final test; test labels must finish by test_end.
    No fitting/transforms occur inside this helper.
    """
    decisions, ends = tuple(map(utc, decision_times)), tuple(map(utc, label_ends))
    start, end = utc(test_start), utc(test_end)
    if len(decisions) != len(ends) or start >= end or embargo < timedelta(0):
        raise ValueError("invalid split dimensions/window")
    if any(a > b for a, b in zip(decisions, ends)):
        raise ValueError("label ends before decision")
    # If labels vary within a date, purge the whole timestamp, not individual rows.
    max_end: dict[datetime, datetime] = {}
    for at, until in zip(decisions, ends):
        max_end[at] = max(max_end.get(at, until), until)
    train = [i for i, at in enumerate(decisions) if at < start and max_end[at] < start - embargo]
    test = [i for i, at in enumerate(decisions) if start <= at < end and max_end[at] <= end]
    return np.asarray(train, dtype=int), np.asarray(test, dtype=int)


def probability_of_backtest_overfitting(returns, *, blocks: int = 8) -> dict:
    """Exact CSCV PBO for a declared synchronized T x N finite return matrix.

    Scores are unannualized sample Sharpe; tied OOS scores use average ranks.
    Constant/degenerate portfolios fail instead of manufacturing Sharpe values.
    Guided search columns must represent independent completed search outputs.
    """
    matrix = np.asarray(returns, dtype=float)
    if (matrix.ndim != 2 or matrix.shape[1] < 2 or not np.all(np.isfinite(matrix))
            or isinstance(blocks, bool) or not isinstance(blocks, int)
            or blocks < 2 or blocks > 12 or blocks % 2 or len(matrix) % blocks
            or len(matrix) // blocks < 2):
        raise ValueError("finite T x N matrix, N>=2, even blocks 2..12 dividing T required")
    partitions = np.split(np.arange(len(matrix)), blocks)
    logits = []
    def scores(indices):
        data = matrix[indices]
        sd = data.std(axis=0, ddof=1)
        if np.any(sd <= 0):
            raise ValueError("constant strategy in CSCV partition")
        return data.mean(axis=0) / sd
    for selected in combinations(range(blocks), blocks // 2):
        inside = np.concatenate([partitions[i] for i in selected])
        outside = np.concatenate([partitions[i] for i in range(blocks) if i not in selected])
        best = int(np.argmax(scores(inside)))
        out = scores(outside)
        rank = float(np.sum(out < out[best]) + (np.sum(out == out[best]) + 1) / 2)
        relative = rank / (matrix.shape[1] + 1)
        logits.append(log(relative / (1 - relative)))
    return {"pbo": float(np.mean(np.asarray(logits) <= 0)), "logits": logits,
            "combinations": len(logits), "trials": matrix.shape[1], "blocks": blocks}


def option_deflated_sharpe(returns, *, independent_trials: int,
                          trial_sharpe_variance: float) -> float:
    """DSR with explicit effective independent trials and unannualized SR variance.

    Pearson kurtosis, sample SD. Independence is a caller assumption, not inferred
    from the number of correlated grid points. No serial-correlation correction.
    """
    values = np.asarray(returns, dtype=float)
    if (values.ndim != 1 or len(values) < 4 or not np.all(np.isfinite(values))
            or isinstance(independent_trials, bool) or not isinstance(independent_trials, int)
            or independent_trials < 1 or not isfinite(trial_sharpe_variance)
            or trial_sharpe_variance < 0):
        raise ValueError("invalid DSR inputs")
    sd = float(values.std(ddof=1))
    if sd == 0:
        raise ValueError("DSR undefined for constant returns")
    sharpe = float(values.mean()) / sd
    centered = values - values.mean()
    m2 = float(np.mean(centered ** 2))
    skew = float(np.mean(centered ** 3)) / m2 ** 1.5
    kurtosis = float(np.mean(centered ** 4)) / m2 ** 2
    normal = NormalDist()
    benchmark = 0.0
    if independent_trials > 1:
        gamma = 0.5772156649015329
        benchmark = sqrt(trial_sharpe_variance) * (
            (1 - gamma) * normal.inv_cdf(1 - 1 / independent_trials)
            + gamma * normal.inv_cdf(1 - 1 / (independent_trials * np.e)))
    denominator = 1 - skew * sharpe + (kurtosis - 1) * sharpe * sharpe / 4
    if denominator <= 0:
        raise ValueError("DSR moment approximation has nonpositive variance")
    return normal.cdf((sharpe - benchmark) * sqrt(len(values) - 1) / sqrt(denominator))
