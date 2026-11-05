# Risk and reporting

Report builders read results; none changes how a backtest runs. Status for all of
them: research diagnostics. A passing check is not risk approval.

## Reports and tear sheets

`result.report()` is the JSON-encodable summary. `reporting.event_tear_sheet`,
`event_tear_sheet_html` and `write_event_tear_sheet_html` build a compact tear
sheet from the same result; `reporting.write_report_html` writes a standalone
HTML report. Research, factor-sweep and execution-sweep reports have matching
`write_*_report_json` / `write_*_report_html` functions (see [API](api.md)).

## Report charts

`equity_curve_charts` builds inline SVG charts of equity, drawdown, long/short
exposure and the return distribution, for `report_html` to embed.

```python
from cortexbacktest.reporting import equity_curve_charts, write_report_html

curve = result.performance.equity_curve()
write_report_html("run", result.report(), "out/report.html",
                  charts=equity_curve_charts(curve))
```

- The generator needs no plotting library and no script; the same input gives
  the same SVG bytes.
- Reports use a light color scheme only. They do not follow the viewer's dark
  mode, and a host page cannot recolor them.
- Charts have no zoom or hover. A consumer that needs interaction can render the
  report JSON itself.
- Fragments passed in `charts` are embedded as they are. The API does not
  sanitize HTML.
- The input can be a Polars frame or a mapping from column name to sequence. A
  mapping without a column skips that chart; a Polars frame without a column can
  raise `ColumnNotFoundError`.
- A curve with fewer than two valid points shows `not enough data`. A constant
  equity curve is still drawn; a constant-return histogram shows the message.
- Each series drops `None` and non-finite values on its own, and the x axis
  spaces the kept points evenly. Date labels come from the first and last
  values of the original series. The axis is not a true time axis, so the
  caller must handle gaps.

## Returns with external cash flows

`reporting.cashflow_performance(valuations, flows, *, currency,
coverage_complete, as_of, flow_timing="end", allow_zero_capital=False)` reports
time-weighted return (exact only when every external flow sits on a NAV
boundary), Modified Dietz (labelled as an approximation) and money-weighted
return. Multiple IRR roots are reported, not resolved by choice. At most 128
dated flows are accepted; histories under 365 days are not annualized.
`coverage_complete=False` gives "insufficient evidence", never zero flows.
Mixed currencies must be converted upstream. `allow_zero_capital=True` keeps
zero-capital intervals as null returns.

`adapters.account_cashflows.account_cashflow_performance` applies this to a file
in the `account-report-v1` contract, after checking its schema and SHA-256.

## Market risk model

`reporting.risk_model.estimate_market_risk(returns, *, benchmark, fit_end,
model_known_at, currency, input_sha256, model_id)` fits a daily single-market
beta model.

Inputs: complete `timestamp`, `symbol`, `known_at`, `return` rows; stable instrument
IDs; an explicit benchmark, currency, fit end and model availability; the original
input digest; and a model ID.

- At least 60 common one-day return observations are required. No missing-value fill,
  duplicate sample, intraday grid, constant market or future data.
- OLS with an intercept. `F` is the sample market variance (ddof 1). Each `D` is the
  residual sum of squares divided by `N − 2`. `D` is diagonal.
- No annualization and no shrinkage. The output has the same shape as the
  style risk model (`estimate_style_risk`), so `factor_risk_decomposition`
  below can read it.

## VaR backtests

`reporting.var_backtest(returns, forecasts, *, confidence=0.99,
significance=0.05)` and `reporting.risk_calibration.historical_var_calibration`
(rolling empirical VaR from an equity curve).

Inputs are Polars frames.

- Observations: `timestamp`, `period_start`, `return`.
- Forecasts: `timestamp`, `forecast_at`, `var`.
- All clocks must be aware and non-null. Timestamps must be unique and match exactly
  between the two inputs. There is no inner join and no dropped missing values.
- Return intervals must be contiguous and positive.
- A forecast must already be known at or before `period_start`, not only before the
  ending close. This catches intraperiod lookahead in the supplied clocks. The caller
  still verifies forecast provenance and the declared confidence.

VaR is a finite, nonnegative loss fraction in the same units as the simple return. An
exception is `return < −var`, strictly, with no hidden rounding tolerance. Values
near the threshold need upstream precision rules. `returns_from_equity` can supply
returns; callers attach explicit interval starts and never invent a missing first
return.

**Tests.** Kupiec LR tests unconditional coverage at the declared exception
probability. Christoffersen independence uses the transition counts n00, n01, n10,
n11. Conditional coverage is the sum of the two LR statistics, with two degrees of
freedom. If an origin state is unobserved or too few observations exist, the
independence and conditional p-values are null with a reason, never a pass. Chi-square
tails are computed with `erfc`/`exp`, without SciPy. The tests have limited power in
small samples.

## Group attribution and factor risk

Functions in `cortexbacktest.reporting`. Real equity style or index
attribution also needs point-in-time characteristics, membership, benchmark
weights and FX that the caller must supply.

### `group_attribution(groups, observed_returns, currency, benchmark_id)`

Inputs are Polars frames.

- Group rows: `timestamp`, `period_start`, `group`, `weights_at`,
  `classification_at`, `currency`, `portfolio_weight`, `benchmark_weight`,
  `portfolio_return`, `benchmark_return`.
- Observed returns: `timestamp`, `period_start`, `currency`, and both returns.
- Returns are decimal fractions in the same currency and interval. Use beginning
  weights including explicit cash. Every group and return must be supplied,
  including explicit conventions for a side with zero group weight. Nothing is
  inferred as a zero return. Signed weights are allowed; each side must sum to one.

Timestamps are aware, kept at nanosecond precision. Intervals must be contiguous,
unique and covered exactly by both inputs. Weights and classification must be known
by interval start. Missing groups, incomplete data, currency mismatch, future clocks
and mismatched observed total returns are rejected.

Fees and cash must be inside the group contributions. Gross group returns are never
compared with a net observed NAV silently; the independent observed returns exist to
detect that mismatch.

**Brinson-Fachler effects** per group, with `wp`/`wb` the portfolio and benchmark
weights and `rp`/`rb` the group returns:

- Allocation `A = (wp − wb)(rb − Rb)`
- Selection `S = wb(rp − rb)`
- Interaction `I = (wp − wb)(rp − rb)`

Single-period effects reconcile to `Rp − Rb`. Carino linking scales each period by
`kt/k`, where `k = (log(1+Rp) − log(1+Rb)) / (Rp − Rb)`. `log1p` keeps near-equal
returns numerically stable; the equal-return limit is `1/(1+R)`. Linked effects
reconcile to the difference of compounded returns. Period or compounded returns at
or below −100 %, or unrepresentable wealth, are rejected.

The report keeps original effects, linking scales and residuals. It never plugs a
residual into an effect. Ordered execution-gap attribution is a separate method.

### `standardise_style_exposures(characteristics, definitions, ...)`

Named definitions map each factor to signed raw characteristics. No proprietary
factor definition is asserted.

Per-date cross-sections use linear-quantile winsorization (default 2.5 %),
positive-cap-weighted centring, and equal-weight sample standard deviation.
Composite styles average signed components weighted by the sum of absolute
coefficients. Constant, missing or nonfinite characteristics, caps and incomplete
schemas are rejected; none becomes zero. The default minimum cross-section is 10.

Output `known_at` is the latest member's availability, because standardization needs
the whole cross-section. Callers own the historical universe, versioned financial
inputs and model identity.

### `portfolio_style_exposure(weights, exposures, model_id)`

Both frames need `timestamp`, `symbol`, `known_at`. Exposures have named factor
columns; weights have `weight` and `benchmark_weight` and must include explicit cash.
Exact asset/time coverage is required.

Reports portfolio, benchmark and active exposures, plus first, last, change, mean,
sample standard deviation and maximum step (drift) for the same declared model. One
observation gives undefined deviation and step, not zero. Drift is descriptive, not a
significance test or trade signal.

### `factor_risk_decomposition(weights, exposures, covariance, specific_variances, as_of, model_known_at, fit_end, model_id, currency, horizon)`

One dated snapshot with complete asset and factor coverage. Model fit must end by the
model availability, which must be no later than `as_of`; all input clocks must also
be available. Specific variances are explicit, finite and nonnegative. Cash needs an
explicit zero exposure and variance when the model assumes it.

With `V = bᵀFb + Σ wᵢ²Dᵢ` and `b = Xᵀw`:

- Factor Euler contribution: `bₖ(Fb)ₖ / √V`
- Asset contribution: `wᵢ(XFb + Dw)ᵢ / √V`

Negative hedge contributions are allowed. Asset contributions, and factor plus
specific contributions, each reconcile to total volatility. Zero-risk shares are
null. Covariance must cover the exact factors, be symmetric and positive
semidefinite. Only relative 1e-12 roundoff in symmetry or eigenvalues is accepted,
and its numerical adjustment is reported. Materially indefinite models are rejected.
Specific covariance is explicitly diagonal. Horizon and currency are declared; there
is no implicit annualization.
