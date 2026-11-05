# Options research and backtesting (experimental)

`cortexbacktest.options` sits next to the OHLC event engine. It reuses the
settlement calendar and code fingerprints, and does not push theoretical option
prices through the OHLC fill model. Pricing and statistics are implemented here;
no third-party code is copied.

## Entry points

```bash
.venv/bin/python examples/option_research_demo.py --output /tmp/cortex-options-demo
.venv/bin/python -m pytest -q tests/test_option_pricing.py tests/test_option_backtest.py tests/test_option_research.py
```

The example is a clearly labelled synthetic short straddle with a daily, lagged
delta hedge of the options held. It writes five spread scenarios, fills, net value
and data/code fingerprints. Its synthetic calendar includes weekends; it is not an
exchange calendar or a reproduction of historical returns. Real data must come
with the target market's sessions.

```python
from cortexbacktest.options import (
    OptionBacktestConfig, ScheduledOptionTargets, run_option_backtest,
)

# contracts: a sequence of OptionContract
# frames: OptionMarketFrame values, strictly increasing in UTC
# decision_time must be a frame time; 0 closes the position, a missing symbol sends nothing.
template = ScheduledOptionTargets(tuple(contracts), {decision_time: {option_symbol: -1}})
result = run_option_backtest(contracts, frames, template.strategy(),
                            config=OptionBacktestConfig(data_id="vendor/version/hash",
                                                        strategy_id="strategy/version"))
print(result.summary())
```

## Data sources

`cortexbacktest.options.chains` loads real option chains: it normalizes DoltHub,
Kaggle SPY and OptionsDX tables to one long table, validates and cleans it, writes
Parquet partitioned by `underlying=/year=`, and rebuilds the `OptionContract` /
`OptionMarketFrame` values this contract needs. Field mapping, cleaning rules and
non-goals are in [option-chain-data-sources.md](option-chain-data-sources.md).

## Data and time contract

- `OptionContract`: unique symbol, underlying, call/put, strike, expiry,
  multiplier, exercise style, cash or physical settlement, account currency.
  `last_trade_at` can be earlier than `expiry`; trading closes at that time.
  `expiry` is the lifecycle expiry instant and is never inferred from a month or a
  string. All timestamps are time-zone aware.
- `OptionQuote` serves both the underlying and options:
  `event_time <= available_at <= frame.timestamp`, with mid from bid/ask. Prices
  and features must be available at `available_at`; open interest or daily volume
  published after the close must not be joined earlier. `bid_size` / `ask_size`
  are displayed executable sizes (contracts for options, shares for the
  underlying), not daily volume.
- `option_frames_from_polars` takes timestamp, symbol, event_time, available_at,
  bid, ask and optional fields, and rejects future or duplicate quotes. It does not
  infer settlement prices; lifecycle events are attached to frames separately.
- A strategy receives only the current frame and a read-only account view.
  Strategy closures and external feature caches can still leak the future and need
  their own review.
- A target set at time t fills on the next frame, from a quote observed strictly
  after t. Without a quote or enough size on that frame the order is IOC-rejected
  or partially filled and the rest is cancelled; a target on the last frame is
  recorded as `end_of_data`.
- Displayed sizes are required by default. `require_sizes=False` is an explicit
  "assume enough liquidity" scenario. Liquidity is never made up from future
  volume, and the size of one quote time cannot be consumed twice.
- A holding with a missing or stale price fails the run; positions are never
  silently dropped, filled at a theoretical price or closed at intrinsic value.
  Zero-bid and wide-spread entry filters do not block real risk reduction, which
  still needs a timely quote and size.

## Pricing models and units

`black_scholes` is European BSM with a continuous dividend yield, `black76`
prices European premiums on a positive forward, and `crr_price` is a CRR tree for
American or European options, vectorized per level with O(N) memory and O(N²)
work; an invalid risk-neutral probability raises. A zero-volatility American
option still compares early exercise at every node.

`options/numerical.py` adds methods for early exercise and path dependence, under
the same conventions:

- `leisen_reimer_price`: odd steps, Peizer-Pratt probabilities, European error about 1/n².
- `trinomial_price`: Boyle trinomial tree.
- `crank_nicolson_price`: log-price grid; the first two steps are fully implicit
  to smooth the payoff kink; American by payoff projection.
- `monte_carlo_price` / `asian_price`: exact GBM paths with antithetic variates;
  returns price and standard error.
- `longstaff_schwartz_price`: American via cubic regression of in-the-money
  continuation values; it is a Bermudan with `steps` exercise dates, so slightly low.

Tests cross-check BS closed forms, a 20,000-step CRR (standard at-the-money
American put 6.0903) and exact Bermudan tree values. Option backtest fills still
use real quotes only; these methods estimate Greeks, early exercise and
assignment risk.

`options/calculator.py` `analyse` prices one option with every model and adds BS
Greeks, an optional implied volatility, value and Greek curves against the
underlying price, and a P&L grid of underlying price × time to expiry
(Leisen-Reimer for American, BS for European). `cortexbacktest option-price` reads
a JSON request on standard input, so a caller can use it without importing the
library.

`options/chain_view.py` `chain_view` shapes one underlying's chain for one session
from an `OptionChainStore`: days to expiry and at-the-money IV per expiry (the
term structure), calls and puts by strike for a chosen expiry with Greeks, the
risk-neutral in-the-money probability N(d2) from each contract's own IV (zero rate
and dividend), the expected move from the at-the-money straddle mid, put/call
open-interest and volume ratios, and the 30-day at-the-money IV rank and
percentile against the previous year. `cortexbacktest option-chain` reads a JSON
request on standard input.

`bsm_greeks` units: delta per unit of the underlying, gamma per unit price change,
vega and rho per 1.0 change in volatility or rate, theta per year. The caller
converts to "per 1%" or "per day". Zero-time and zero-volatility Greeks are
rejected at the singularity instead of returning false precision.

`implied_volatility` inverts European BSM only, with bisection, bracket and
no-arbitrage bound checks; without a solution it does not fall back to historical
volatility. American IV and Greeks should come from external data with a declared
model; this version has no American IV inversion and no discrete cash-dividend
tree. Black-76 pricing does not mean futures-option daily settlement is supported.

## Trading costs

The cost model is a plugin (`OptionCostModel`). The default `ResearchOptionCosts`
reproduces the original behaviour (the full half-spread plus 0.65 per contract);
every other component is off until enabled:

```text
cost = option spread + underlying spread + commission + slippage + market impact + hedge turnover
```

- **Spread**: `spread_fraction` is 0 at mid and 1 at the far side.
  `run_option_scenarios` sweeps it (with a plugin, the plugin's own field).
- **Commission**: `option_commission` per contract, `option_commission_bps` on
  premium notional, `option_minimum_ticket` per order; the underlying uses
  `stock_commission_bps`.
- **Slippage**: `option_slippage_ticks × tick_size`; the underlying uses
  `stock_slippage_bps`. End-of-day chains cannot measure slippage, so this is a
  declared value, not an estimate.
- **Impact**: `impact_coefficient × half-spread × sqrt(size / impact_reference)`,
  the standard square-root shape. End-of-day chains have no book depth, so
  `impact_reference` is the size at which impact equals `impact_coefficient`
  half-spreads, also declared. The default 0 is honest for one-contract research
  and wrong for scaled-up research.
- **Hedge turnover** is not a parameter: it is the fills the rebalancing
  frequency produces, each priced by this model. Hedge frequency has a first-order
  effect on net returns, which is visible only when every fill is charged.

Every deviation of the fill price from mid goes to the `option_spread` /
`stock_spread` attribution items (execution cost relative to mid), and the account
identity holds. The run manifest records `cost_model` and `cost_parameters`.

With a plugin, `spread_fraction` / `option_fee` / `stock_fee_bps` on the config
must keep their defaults, or the config raises: two places to set one number is how
a run ends up with a cost nobody chose.

## Lifecycle, cash and margin

`OptionLifecycleEvent(quantity=None)` is an expiry and needs the official fixing
and its availability time; a position reaching expiry without an event raises.
Cash-settled expiry pays positive intrinsic value. Physical expiry exercises by
default when intrinsic value is positive and at least 0.01; the threshold is
configurable, or `exercise=False/True` overrides it. OCC exception rules allow
contrary instructions, so the default does not predict real assignment.

American early exercise and assignment need explicit events, with a quantity of
the position's sign and no larger than it. Model-optimal exercise does not create
assignments. A physical call delivers `quantity * multiplier` shares against the
strike, a put the reverse, symmetric for shorts. Contracts delivering the same
underlying quantity are supported; split adjustments, multi-asset or cash-basket
delivery, special corporate actions and exchange auto-liquidation are not.

Cash settlement goes to economic cash; a positive cash flow is unsettled before
`settlement_at` and cannot serve as collateral or earn interest before then.
Immediate settlement is only the research default; real runs use
`SettlementModel.trading_days(...)` or an explicit `settlement_at` per event.
Physical delivery changes economic shares at once; there is no separate pending
security inventory. Financing accrues ACT/365 by interval and cash release time,
plus stock borrow fees and explicit per-share dividends.

Two margin models ship, both recomputed every frame, neither certified Reg-T or
SPAN; a plugin implementing `OptionMarginModel.requirement` replaces them.

`ResearchOptionMargin` (default): full premium for longs, premium plus a risk term
for naked shorts, a fraction of absolute market value for the underlying, and
**no netting**. Every short leg counts as naked, so a protective leg raises the
requirement: a SPY 390/380 put spread needs 7,800 against a real maximum loss of
800. It suits naked and single-leg research, not spreads.

`RegTOptionMargin` nets by the Reg-T stock-option account rules and recognizes four
structures: covered stock (long stock covering short calls, short stock covering
short puts); verticals (same underlying, expiry and type; credit spreads charge the
strike width, debit spreads the long premium); iron condors (one expiry, only the
wider side charged, because the underlying cannot be outside both); and short
combinations at one expiry (straddles and strangles charge the larger naked side
plus the other side's premium). Remaining shorts are naked; unpaired longs are
charged full premium (a long option has no loan value). Pairing order is fixed by
expiry, strike and symbol, independent of position insertion order.

It is **not** portfolio margin: no SPAN, no cross-expiry or cross-underlying
offsets, no delta netting. A real broker gives a different number; the goal is to
charge defined-risk positions near their real maximum loss rather than as naked
shorts. Insufficient initial collateral rejects the trade. A maintenance shortfall
tries real-quote liquidation in a stable order; without liquidity the positions
stay open with a breach record. Multi-leg orders fill leg by leg and can briefly
leave naked risk; atomic combination fills are not promised.

## Returns and experiments

Account identity:

```text
net_pnl = option_market_pnl + stock_market_pnl
          - option_spread - stock_spread - option_fees - stock_fees
          + financing + dividends
```

Market P&L is for the quantities actually filled; physical exercise moves
intrinsic value between the option and underlying books at the fixing. It is not
"what the strategy would earn without costs" and not a delta/gamma/vega Taylor
decomposition. `run_option_scenarios` builds a new strategy, account and
deep-copied config for each spread fraction and compares mid, 25%, 50%, 75% and
the full bid/ask with everything else fixed. When financing and positions change
between scenarios, the return difference is not just the summed spread cost.

`summary()` reports total return, maximum drawdown, per-frame 95% tail loss and
margin, without annualizing frames of arbitrary frequency. Frame returns that are
undefined after net value turns non-positive are not filled in. Deltas of different
underlyings are not added, so the net-delta total is empty; call
`delta_hedge_targets` per underlying.

## Research definitions and out-of-sample tools

- `goyal_saretto_signal` is log(HV) − log(mean(call IV, put IV)). The historical
  volatility helper uses the window it is given and does not guarantee a full
  12 months. Deciles, entry lag, static hedges and capital normalization of the
  paper are the researcher's to set up; no full replication is claimed.
- `select_option` picks the closest target DTE, then ATM or delta distance, then
  strike and symbol as deterministic tie-breaks, within the DTE range and quote
  filters.
- `ScheduledOptionTargets` sends targets at given times and can hedge the actual
  position on a schedule or a threshold. The first hedge comes after the option
  fill, on the next frame; it is not a simultaneous risk-free hedge.
- `purged_option_split` splits whole timestamp cross-sections; training labels
  must end strictly before `test_start − embargo`, and test labels by `test_end`.
  Scaling, parameter selection and trading history must still be fitted per fold.
- White's Reality Check and Hansen's SPA are in `cortexbacktest.research.validation`.
  The input is a T×M matrix of per-period performance relative to a benchmark and
  must include every candidate searched, abandoned ones too; the null is that no
  strategy in the family beats the benchmark. The null distribution uses the
  Politis-Romano stationary bootstrap, keeping serial correlation; `block_size=1`
  is iid. SPA also studentizes and recentres and gives `p_lower` / `p_upper`; a
  large gap means the conclusion depends on the recentring threshold.
  `sharpe_haircut` is the Harvey-Liu haircut from a non-annualized Sharpe and the
  trial count, treating the result as the most significant (Holm then equals
  Bonferroni; BHY adds a harmonic factor).
- CSCV PBO needs a synchronous, complete T×N candidate return matrix, an even
  number of 2 to 12 equal blocks and exact combination enumeration; a constant
  partition is rejected. OOS ties take average ranks, IS ties keep column order.
  Adaptive search uses each complete search result as a column; deliberately poor
  intermediate candidates must not pad it.
- DSR needs the effective number of independent trials and the variance of the
  non-annualized trial Sharpes, with Pearson kurtosis. There is no automatic
  correlation estimate or serial-correlation correction, and no "passed PBO/DSR"
  without its inputs.

## Scope

This is a runnable options research kernel, not a whole-market production trading
simulator. Not included: data purchasing, automatic survivorship repair, American
assignment probabilities, volatility-surface or Heston calibration, LSM exotics,
portfolio margin, multiple currencies, complete corporate actions, options-account
checkpoint recovery and automatic parameter wiring.

Next, in order: licensed historical chains, settlement fixings and calendars for
the target market, with point-in-time fields and adjusted contracts audited; then a
frozen out-of-sample replication; then American discrete dividends and IV,
portfolio margin and combination fills; then Heston, LSM and ML as evidence needs.

Timing notes: `frame.dividends` books economic dividends on the shares held at the
start of the frame; it is not a pay-date cash ledger with ex, record and pay
dates. Split frames around an ex-dividend event that involves assignment in the
order the events take effect. A margin plugin's `requirement` must be a
deterministic function without side effects.

## Volatility strategy library (`options.strategies`)

A strategy is **structure + parameters + volatility gate**, with no underlying:
one candidate is the same identity on SPY, AAPL or any stored underlying (the
`candidateId` hash excludes the underlying).

- Structures: `short_put`, `bull_put_spread`, `short_strangle`, `iron_condor`,
  `long_straddle`. Legs are placed by delta on one expiry; protective legs must be
  further out of the money.
- Parameters: `dte` (30/45), short-leg `delta` (0.16/0.30), protective `wing`
  (0.05/0.10), `profitTake` (hold to expiry, or close at half the premium).
- Gates, using only data up to the day: `iv_hv` (30-day at-the-money IV ÷ 21-day
  realized volatility) and `iv_rank` (at-the-money IV's position in the previous
  252 sessions). Short-volatility structures use "above", the long straddle "below".
- Size: each leg's underlying notional as a fraction of equity (`--exposure`), so a
  candidate means the same on a 50-dollar and a 500-dollar underlying.
- Splits: `--splits` (`symbol,ex_date,factor`) adjusts only returns and realized
  volatility; strike comparisons and sizing use unadjusted traded prices.


**DoltHub single-stock chains cannot drive holding backtests.** They are sampled,
not continuous: Mondays, Wednesdays and Fridays only, three rolling expiries a day,
about 22 strikes per expiry that change daily. A median AAPL contract appears on 3
sessions (22 for the Kaggle SPY chain) and 21% appear once. Holding a position
needs a quote for the same contract every session, and the engine refuses to value
a price that does not exist, so those underlyings leave 3–4 trades. Filling gaps
with theoretical prices would break the rule of filling on real quotes only.
DoltHub still suits cross-sectional work (one day's IV surface, skew). Testing
across underlyings needs continuous per-contract end-of-day chains, such as
OptionsDX (some symbols and years free, manual download) or paid sources.

