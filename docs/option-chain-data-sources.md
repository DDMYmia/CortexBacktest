# Public option-chain data (DoltHub / Kaggle / OptionsDX)

`cortexbacktest.options.chains` is the boundary between raw option-chain files and
the `options` research kernel. It normalizes each vendor's format to one long
table, reports and filters the known quote defects of these datasets, writes
hive-partitioned Parquet, and rebuilds the `OptionContract` / `OptionMarketFrame`
values the engine needs.

The kernel accepts only strict point-in-time objects (see
[options-backtesting.md](options-backtesting.md)), while public datasets are
CSV/Parquet tables. This layer converts between them and **never supplies what the
data does not have**: no inferred settlement prices, no inferred publication
times, no daily volume posing as executable size.

## Unified schema

`CHAIN_SCHEMA`, one row per contract per observation time:

```text
quote_date          time-zone aware UTC instant (not a date)
underlying          upper-case underlying symbol
underlying_price    underlying price; DoltHub has no such column and needs a join
expiry              time-zone aware UTC expiry instant
dte                 recomputed as calendar days between expiry and quote_date; the vendor column is ignored
strike
right               CALL / PUT
option_symbol       21-character OCC symbol, e.g. SPY   220715P00390000
bid ask mid
implied_volatility delta gamma theta vega rho
volume open_interest
```

`quote_date` and `expiry` are **instants**, not dates. An end-of-day file has only
a date; which instant it stands for is the researcher's assumption, given
explicitly by `ChainTimePolicy`, not a fact in the file.

```python
ChainTimePolicy(quote_time=time(20, 0),       # UTC; 16:00 ET during daylight time
                expiry_time=time(20, 0),
                publication_lag=timedelta(0)) # event_time -> available_at
```

The default 20:00 UTC equals 16:00 ET only during US daylight time. Data that
spans winter months, Cboe's 15:45 ET snapshot or AM-settled index options need
their own values. `publication_lag` is the delay between observation and the time
an order can use it; it becomes both `available_at` and the frame timestamp.

## Three sources

| | DoltHub `post-no-preference/options` | Kaggle SPY EOD 2010–2023 | OptionsDX |
| --- | --- | --- | --- |
| Format | SQL database / API JSON (long) | One Parquet per year (**wide**, bracketed headers) | One CSV/TXT per month (wide) |
| Coverage | Thousands of US stocks and ETFs | SPY only | Per-symbol downloads, some years free |
| Starts | **2019-02-09** (measured) | 2010 | Depends on symbol and year |
| Underlying price | **No**, needs a join | Yes | Yes (`UNDERLYING_LAST`) |
| IV / Greeks | Yes (IV in `vol`) | Yes | Yes |
| Source name | `"dolthub"` | `"kaggle_spy_eod"` | `"optionsdx"` |

They complement each other: **Kaggle SPY for long history, DoltHub for
cross-sectional breadth and recent data**, OptionsDX for the SPX/QQQ/single-stock
years that happen to be free.

Column names match after normalization (`_key`): case, spaces and brackets do not
matter, so OptionsDX `[C_BID]` equals `c_bid`. Aliases match in order; the first
hit wins.

- The Kaggle SPY EOD dataset is published on Kaggle but uses the OptionsDX wide
  format (`[QUOTE_DATE]` / `[C_IV]` / `[P_IV]`…). `"kaggle_spy_eod"` is therefore a
  wide source and defaults to `underlying="SPY"` (the files have no symbol column);
  an explicit `underlying=` still wins.
- DoltHub's `vol` is implied volatility and is read as IV only for `"dolthub"`. In
  community exports `vol` can be volume, so `"kaggle_spy_eod"` and `"long"` ignore
  that alias.
- For an unknown header, name the column with `overrides={"bid": "bid_px"}` or
  build an `OptionChainSource`; both take precedence over built-in aliases.
  Per-leg fields of wide tables cannot use `overrides` (they would hit both the
  call and the put leg); rename the column or build a source.

## Download

`scripts/fetch_option_chains.py` needs only the standard library and this engine,
and covers fetch → normalize → clean → store:

```bash
# DoltHub public SQL API, no API key; fetched one day at a time (see below)
.venv/bin/python scripts/fetch_option_chains.py dolthub --underlying SPY \
    --start 2024-01-02 --end 2024-01-31 --root data/options \
    --prices data/bars/spy.csv          # DoltHub has no underlying price; supply one

# Files already downloaded
.venv/bin/python scripts/fetch_option_chains.py file data/raw/spy_eod_2020.parquet \
    --source kaggle_spy_eod --root data/options
.venv/bin/python scripts/fetch_option_chains.py file data/raw/spy_2020_january.txt \
    --source optionsdx --underlying SPY --root data/options
```

Getting the raw files is the vendor's business: `kaggle datasets download -d
<slug> --unzip` for Kaggle, `dolt clone post-no-preference/options` for the whole
DoltHub database, and a ZIP from the OptionsDX site.

### Kaggle SPY EOD, measured (full import, 2026-09-06)

`kagglehub.dataset_download("dudesurfin/spy-options-eod-volatility-surface-2010-2023")`
gives 14 yearly Parquet files (486 MB compressed, 569 MB uncompressed) whose
33-column headers match the `"kaggle_spy_eod"` wide layout. After a full import:

```text
18.937M rows (after splitting wide into long)
  -> 18.557M rows stored, 0.380M removed (2.0%)
     null_key                  23,368
     nonpositive_quote        106,588
     unrepresentable_feature  217,116   negative IV; OptionQuote rejects it
     crossed                   33,340
3,500 sessions, 2010-01-04 .. 2023-12-29
368k distinct contracts, about 553 MB on disk
```

| Finding | Consequence |
| --- | --- |
| **About 4 sessions missing per year** (2019 has 248; 2019-06-07 has no data at all) | Sessions cannot be assumed contiguous |
| **49 of 1,572 expiries have no matching session** | Those contracts cannot settle at their official expiry; research must skip them or invent a fixing |
| **217k rows with negative IV** | A data error, not deep out-of-the-money; `OptionQuote` rejects it, so normalization treats it as an error, not a warning |

The second finding costs real backtests: over 2010–2023,
`examples/spy_short_put_study.py` skipped 20 of 176 candidate entries for a missing
expiry and dropped 16 for quote gaps beyond the tolerance, about 20% in all. Report
this number in any study built on this dataset.

`[QUOTE_READTIME]` shows a **16:00 ET snapshot**. Fourteen years cross daylight
time changes, so import with
`ChainTimePolicy(quote_time=time(16,0), zone="America/New_York")`
(`scripts/ingest_kaggle_spy.sh` does); a fixed UTC time puts half the sample off
by an hour.

### DoltHub API, measured (2026-09-06)

| Fact | Consequence |
| --- | --- |
| The default branch is `master`, not `main` | `main` fails with `branch not found` |
| One response holds at most 1,000 rows; a truncated one has status `RowLimit`, not `Success` | Treating `RowLimit` as failure stops at the first page |
| `date = 'X'` hits the primary-key prefix and returns in about 1 s; `date BETWEEN … AND act_symbol = …` hits the server query deadline (`context deadline exceeded`) | The script fetches **one day** at a time and pages within the day |
| A cold query takes about 80 s per day, a warm one about 1 s | A month takes about 40 minutes |
| Coverage 2019-02-09 to 2026-09-04 (on the check date) | |
| About 3 SPY expiries and 100–160 rows per session | **Not a full chain**, a thinned cross-section |
| Whole days are missing (2024-01-09 has no SPY rows) | Handle missing days explicitly |
| Only about 1/3 of contracts are quoted on two neighbouring days (2024-01-08 → 01-10: 38 shared contracts of 106 and 104 rows) | Holding strategies must check contract continuity; the engine rejects with `missing_quote` instead of inventing a price |

The last point matters most: DoltHub suits **cross-sectional** research (many
symbols on one day). For time-series backtests that hold positions, its contract
coverage is broken; use Kaggle SPY for long, continuous history.

## Validation and cleaning

`validate_option_chain` splits problems into two kinds, and the split is part of
the interface:

**Errors, rows the engine cannot represent**: empty key columns, duplicate
contracts, `right` not CALL/PUT, strike ≤ 0, bid < 0, ask ≤ 0, ask < bid
(crossed), expiry before the quote date. `OptionChainStore.write` and
`option_chain_to_frames` validate first and refuse data with errors.

**Warnings, known in public datasets and a defect only for some research**: zero
bid, very wide spreads, IV out of bounds, delta out of bounds or with the wrong sign
for its right, missing underlying price, large put-call parity residuals. They do
not stop the flow: a zero bid on a deep out-of-the-money contract is a real market
state, and a parity break on a violent day can be a real liquidity event.

The parity check tests whether the residual of `(C - P)` against
`(S·e^{-qT} - K·e^{-rT})` exceeds `parity_tolerance × S`, with `parity_rate` and
`parity_dividend_yield` defaulting to 0. **It is a coarse screen** for vendor
errors such as mismatched legs, a stale side or a wrong underlying price, not a
term-structure calibration.

`clean_option_chain` removes rows rule by rule per `ChainFilter` and returns the
count each rule removed; counts go to the first matching rule, so they add up to the
total:

```python
cleaned, removed = clean_option_chain(chain, ChainFilter(
    drop_zero_bid=True, max_relative_spread=1.0, min_open_interest=0,
    implied_volatility_bounds=(0.01, 5.0), dte_range=(0, None)))
# removed == {"zero_bid": 1183, "wide_spread": 402, ...}
```

Record the removed counts with the research: filter strength changes the sample of
short-put or iron-condor strategies a lot.

## Storage

`OptionChainStore` partitions by underlying and year, the same layout as the tick
storage of `HistoricalDataStore`, for the same reason: filtering by symbol and date
skips whole directories instead of reading fourteen years of SPY for a one-month
question.

```text
<root>/underlying=SPY/year=2019/part-<id>.parquet
```

`write` only adds partition files and never rewrites files on disk, so importing a
vendor file costs O(rows in the file). De-duplication belongs to
`clean_option_chain` before the write, not to storage. `scan()` returns a
LazyFrame; `underlyings`, `start` and `end` push down to both partition and row
filters.

## Back into the engine

```python
from cortexbacktest.options import (
    OptionBacktestConfig, OptionChainStore, QuotePolicy, attach_lifecycle,
    expiry_events_from_underlying, option_chain_to_frames, option_contracts_from_chain,
    run_option_backtest,
)

chain = OptionChainStore("data/options").scan(
    underlyings=["SPY"], start=start, end=end).collect()
contracts = option_contracts_from_chain(chain)          # American, physical, 100 shares
frames = attach_lifecycle(
    option_chain_to_frames(chain, underlying_half_spread=0.01),
    expiry_events_from_underlying(chain, contracts))
result = run_option_backtest(contracts, frames, strategy,
                             config=OptionBacktestConfig(require_sizes=False))
```

`option_chain_to_frames` adds one underlying quote per observation time
(`underlying_price ± underlying_half_spread`): valuing option positions and margin
needs the underlying price. Use `include_underlying=False` only for runs that hold
nothing.

`option_contracts_from_chain` defaults to US listed stock and ETF options
(American, physical, 100). Index options (SPX, XSP, VIX) are European and
cash-settled, and adjusted contracts have non-standard multipliers; none of that can
be inferred from a chain file, so pass it explicitly.

## Non-goals

- **No sizes.** End-of-day chains publish no executable size, and daily volume is
  not size, so `bid_size` / `ask_size` are empty and runs need
  `OptionBacktestConfig(require_sizes=False)`: an explicit "enough liquidity"
  scenario, not proof of frictionless fills.
- **`expiry_events_from_underlying` is an approximation.** The engine needs an
  explicit fixing at expiry, and public end-of-day chains have no settlement price,
  so the function uses the chain's own underlying close on the expiry date. That is
  not the official settlement (PM-settled stock options settle on a close, not
  necessarily this vendor's; AM-settled index options settle on the opening auction).
  Research can use it and must say so. A contract whose expiry has no row in the
  chain gets no event, and the backtest fails at delivery instead of expiring at an
  invented price.
- **No data repair.** Validation reports, cleaning removes rows; nothing
  interpolates, extrapolates or fills bid/ask with theoretical prices.
- **No survivorship, corporate-action or adjusted-contract handling.**
  Non-standard multipliers after splits and adjusted symbols are the caller's job.
- **The Kaggle set is community data, not Cboe's.** Users have reported clear
  call/put IV inconsistencies on violent days in 2018. Publication-grade research
  should refilter and check its sample.

## QQQ, two silent traps, financing and fixings (2026-09-21)

### Free coverage by underlying

| Underlying | Free historical chains | Note |
| --- | --- | --- |
| SPY | Kaggle `dudesurfin/...2010-2023` | 14 years |
| QQQ | Kaggle **`kylegraupe/qqq-daily-option-chains-q1-2020-to-q4-2022`** | See below |
| TSLA / AAPL / NVDA | Kaggle `kylegraupe/*`, same series | |
| **TQQQ / SOXL** | **None** | Neither Kaggle nor DoltHub; paid sources or a QQQ proxy |

DoltHub cannot serve these leveraged ETFs: on two sessions with 1,637 and 625
symbols, QQQ, TQQQ, SOXL and IWM had no data at all, only SPY, AAPL and TSLA, and
SPY had about 148 rows a day, too thin for near-the-money selection. Its value is
breadth, not single-symbol depth.

The QQQ dataset (`--source optionsdx --underlying QQQ`, OptionsDX wide format):

```text
Titled Q1 2020 - Q4 2022; the files hold 2021-01-04 .. 2022-12-30 only
508 sessions, 1,775,749 raw rows -> 3,522,593 stored (after splitting wide into long)
82,893 contracts, about 105 MB
delta complete; implied_volatility 6.2% missing; volume 24.3% missing
open_interest empty in every row
[C_SIZE] / [P_SIZE] displayed sizes present (the SPY set has none)
```

A number for the research record: with default parameters,
`spy_short_put_study.py --underlying QQQ` skipped **14 of 36 candidate entries
(39%)** because the expiry had no matching session in the data.

### Trap 1: vendor CSV spaces turn numbers into null silently

OptionsDX writes a space after each comma (`", 323.590000"`).
`cast(Float64, strict=False)` turns `" 323.59"` into null **without an error**, and
`str.to_datetime` does the same. An import without stripping stores prices,
strikes and Greeks as null and fails somewhere else for no visible reason.
`normalize_option_chain` now strips text columns before converting.

### Trap 2: a threshold on a column the vendor never publishes removes everything

`open_interest` is **empty in both the SPY and QQQ sets** (the OptionsDX wide format
does not publish it, and DoltHub's `option_chain` table has none). Null compares
false, so `ChainFilter(min_open_interest=10)` removes every row and reports it as a
working liquidity filter. `clean_option_chain` now raises on a threshold over an
all-null column and names it, and `validate_option_chain` warns that the source does
not publish `open_interest`. On these sets, filter liquidity on `volume` only.

### Financing rates and fixings

`examples/spy_short_put_study.py` takes two inputs that used to be hard-coded:

```bash
--rates  data/raw/DGS1MO.csv       # FRED observation_date,VALUE; no key:
                                   # https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS1MO
--closes data/raw/spy_closes.csv   # ticker,session,close, used as the expiry fixing
```

**Without `--rates` cash earns 0%**, which badly understates a cash-secured
programme: on the 2010–2023 DGS1MO curve, financing was 14,304 per 100,000 of
capital, more than the option P&L of the same fills (10,200).

There is a new trap the other way: with financing, the per-frame Sharpe of the same
fills went from **0.0235 (p = 0.18) to 0.0475 (p = 0.0061)**, from not significant
to significant, without a single fill changing. The return became a **total
return** instead of an excess return: the financing leg is T-bills, which belong to
the benchmark too. `assumptions` now prints this. To compare the strategy itself,
use the 0% run or raise the benchmark to the same curve.

`RateSchedule` is a step function by date that takes the last point on or before
the date. A date before the curve starts **raises instead of earning 0%**: a
backtest silently run at 0% shows nothing in its result. With `financing=` set,
`credit_rate`, `debit_rate` and `stock_borrow_rate` must stay 0, so one rate has one
source.

`expiry_events_from_prices()` separates the fixing from the chain under test. The
chain's own `underlying_price` comes from the vendor pricing both sides of the
trade; an independent source can at least disagree when one is wrong.
`session_closes()` builds that table from a minute tape, taking the last bar
**before** 16:00 ET, so a 16:05 after-hours print is never the close.

Full sample (2010–2023, 3,197 matching sessions, Kaggle SPY chain against a
minute tape):

```text
mean absolute difference   10.5 cents / 4.4 bp        median 6 cents
maximum                    2.64 dollars (2022-09-30)   27 sessions above 25 bp
2020 worst at 7.85 bp, 2019 best at 3.05 bp
```

The differences are concentrated: of the 27 large ones, 5 are month-ends, and
others are **half days** such as 2011-11-25, 2012-12-28 and 2021-11-26, which have
no 16:00 close, so the vendor's "16:00 snapshot" must differ from the real last
bar. On the 182 expiries actually traded, the largest was 2020-03-16 (a
circuit-breaker day) at 1.76 dollars a share, 176 dollars a contract.

**Changing the source barely moves the total.** The same 140 entries and fills,
with only the fixing changed:

| 2010–2023 | Chain's own close | Independent minute tape |
| --- | --- | --- |
| Option P&L | 10,200.50 | 10,095.50 |
| Stock P&L | 3,345.00 | 3,752.00 |
| Net P&L | 27,403.90 | **27,731.68** (+1.2%) |
| Per-frame Sharpe | 0.0475 | 0.0481 |

One day can differ by three trades' profit, yet fourteen years differ by 1.2%: the
signs mostly cancel. An independent fixing **removes an unverifiable assumption**;
it does not change the conclusion, and it will not rescue a strategy that does not
work.

Neither is the **official settlement price**: stock options deliver by OCC's
settlement-price procedure, not by anyone's close.
