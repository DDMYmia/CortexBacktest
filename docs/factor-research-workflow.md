# Factor research workflow (panel → factors → verdict)

This document records the **order of the steps and the ones that cannot be
skipped**. It is not a conclusion about any factor.

```sh
examples/factor_workflow_demo.py --panel data/panel --output /tmp/factors
```

The output is deterministic: rerunning on the same panel gives a byte-identical
`factor-report.json`, and `data_digest` and `code_digest` go into the manifest.

## Why this order

The workflow grew out of a failure. A first run of eight factors gave a table
sorted by IC that looked like a conclusion. It was not:

- **The top factor by IC had a negative long/short Sharpe.** `momentum_12_1` had
  rank IC +0.017 (IR 1.31, the highest) and a quintile long/short Sharpe of
  **−0.41**. Momentum's relation lives in the middle of the distribution and
  extreme winners crash: rare large losses barely move a rank correlation but
  dominate the mean spread.
- **The top factor by long/short Sharpe broke even at 10.6 bp.** Full costs
  (spread plus impact) for liquid US names are 5–15 bp, so its whole excess sat
  inside the cost uncertainty.
- **The family passed only the loosest test.** White reality check p = 0.042,
  Hansen SPA p = 0.090, Benjamini-Hochberg rejected none of eight, deflated
  Sharpe 0.727 (0.95 needed), Bonferroni-adjusted p = 0.313.

So `run_factor_workflow` cannot return half a result: a `FactorEvaluation`
cannot be built without turnover and the break-even cost, and a set of them
cannot be returned without a family verdict. **A single factor is priced as a
search too**: it is a family of one, not a free hypothesis.

`established` needs **every** correction to agree, not any one of them. In the
run above White alone passed; accepting it would have reported a discovery.

The search family must be complete. When explicit `trials` exceed the candidate
return columns supplied, the diagnostics still apply the full trial penalty, but
`family.complete=false` and the verdict cannot be `established`. Conservative
BH/DSR penalties for omitted trials do not replace the complete return family
that White and SPA need.

`pipelines.factor_workflow.evaluate_return_family` applies the same corrections
to frozen event returns without recomputing factors or events. The caller gives
shared dates without gaps, each candidate's per-period net return difference and
the total number of trials; a zero benchmark answers only the absolute-return
question. `statistically_supported` is not a trading qualification: it does not
replace benchmark consistency, cost sources, frozen evidence, tradability or a
real forward test. The cross-sectional workflow keeps its own breadth, cost and
sign-consistency checks.

## Step 1: the panel

Acquisition recovery is strict: SSH errors, timeouts and empty responses fail
instead of being recorded as absent. An existing file is reused only after it is
read in full and passes field, month and unique-identity checks. New files are
written to a temporary file and published without overwriting an existing one.
Give `--start-month` / `--end-month` from the real source range (for a source that
ends in 2026-03, use `--end 2026 --end-month 3`); missing later months must not
count as success. These checks do not prove that the source is complete or
point-in-time.

`scripts/build_factor_panel.py` aggregates the remote minute store as a stream;
**minute data never lands locally** (about 100 MB a month). That gives one hard rule:

> A feature that needs minute data must be computed during extraction. Adding one
> later means another pass over the whole tape.

Add new features to `minute_features()`; do not plan to derive them downstream.
The panel has 13 fields; `realized_variance`, `max_minute_move`, `rv_per_minute`,
`open_volume_share` and `close_volume_share` cannot be built from daily files.

A common mistake: one-minute returns must use `.over("ticker", "day")`.
Otherwise the first bar of each session inherits the previous close and the
**overnight gap counts as intraday volatility**, which ruins realized-variance
estimates.

**Point-in-time membership comes from the data source, not from this script.**
The research layer recomputes its members from the **previous** month only:

| | Membership | 2015-06 → 2025-06 |
| --- | --- | --- |
| Point-in-time research layer | Top 1,000 by the previous month's dollar volume | **476 of 1,000 left (48%)** |
| `universe_daily.parquet` | `days > 3800` over the whole period, then top 1,500 by dollar volume | **none left** |

The second is an engine throughput benchmark; its script says on line one that it
is deliberately not the point-in-time research layer. Used for factor research,
every long leg would learn that the name survived, and **none of the statistical
defences here can stop that**: they guard against too many trials, not against a
sample of winners.

## Step 2: factors and targets

Three choices matter more than the factor definitions; all are in
`examples/factor_workflow_demo.py`. The example needs a daily panel: the 5/21/252
row windows, intraday variance × 252, entry on the next session and exit on the
session after are all in sessions. Monthly files are only storage partitions.
Weekly or monthly research needs its own features, labels and windows.

1. **Open to open, not close to close.** A factor computed from today's close
   cannot trade at today's close; close-to-close scoring credits a move it could
   not capture.
2. **Liquidity filters apply on the signal date only.** Filtering on the return
   date selects the sample with the future.
3. **The caller builds forward returns; `run_factor_workflow` has no default.**
   The price and the lag are the most important choices of the workflow and do
   not belong in a default argument.

## Step 3: the verdict

`sign_disagreement` in `FactorEvaluation` is a cheap warning: rank IC and the
long/short spread have opposite signs, so the relation lives in the middle and the
tails point the other way. Four of the eight factors in the first run raised it.

`ic_ar1` reports the IC's first-order autocorrelation (−0.077 to +0.048 for the
eight candidates). A value near zero does not prove independence or rule out
higher-order correlation, overlapping labels or correlated returns; it alone does
not justify square-root annualization.

`FactorWorkflowConfig(periods_per_year=...)` sets the observations per year
(default 252 for daily; 52 weekly, 12 monthly). Annual return scales linearly and
Sharpe and IC IR by the square root; the value goes into the in-sample and
out-of-sample manifests. It only converts units: it does not resample, change
windows or labels, or correct autocorrelation. The caller keeps labels,
rebalancing and observation frequency consistent and evaluates overlapping
holding periods separately.

`cost_bps` defaults to 10. It is not a model but **the reader's assertion**, and
it goes into the manifest so a later reader can disagree explicitly.

Label completeness comes first: a valid factor with a missing or non-finite
forward return raises. Dropping nulls would select survivors by whether they still
had quotes later. The example may drop the last two global dates of the
extraction; inner gaps need holding-period quotes or an evidenced terminal return,
not zero-filling or dropping.

`research.labels.attach_forward_labels` builds labels from quotes: signal-date
decisions, complete holding-period quotes and an explicit trading calendar, joined
on the canonical `instrument_id` for exact entry and exit dates, keeping the number
of decision rows. It accepts only prices and dataset identities declared
`total_return_adjusted` and outputs simple total returns with coverage counts;
the declaration does not replace source verification. Missing quotes, invalid
quotes and global right censoring are kept and counted separately; a zero quote is
not evidence of a cash delisting.

## Limits

- **Options and macro data are time series, not cross sections.** Over 4,622
  sessions there are fewer than 20 independent volatility-regime changes, so any
  "when VIX is high" rule has a two-digit effective sample. Use them to condition
  cross-sectional factors, not as factors to rank.
- Five of the eight candidates share a 21-day window. Measure their correlation;
  a shared window alone does not justify fewer trials. Keep every candidate tried.

## Shared dates and out-of-sample

The numbers above are from the first run and were not recomputed with the current
code. `factor_family_v2` aligns candidates on shared valid dates and checks the
cross-section size after dropping each factor's null or non-finite values;
duplicate dates or identities raise. The manifest records each factor's raw, valid
and shared date counts. Each candidate needs its own BH decision, and White is part
of the verdict gate.

Cost counts the target-weight change of both unit-notional legs, initial entry
included. `turnover` is the average one-way turnover of the two legs, and the
daily cost is `4 * turnover * cost_bps / 10000`. It is a research approximation
without weight drift, borrow or final liquidation, not an execution ledger.

```sh
.venv/bin/python examples/factor_workflow_demo.py --panel /path/to/panel \
  --walk-forward --train-sessions 756 --test-sessions 252 \
  --embargo-sessions 0 --output /tmp/factor-oos
```

`run_factor_walk_forward` splits on unique sessions and purges labels that cross
a boundary; the example's horizon 2 covers entry at t+1 to exit at t+2. The example
rejects using a symbol's next record across a missing session as t+1/t+2. The
caller still guarantees a complete session index, causal features and true label
semantics.

The first version fixes every candidate and its direction, with no fitting and no
selection. Preprocessing that needs fitting must fit on the training window;
features fitted on the full sample do not remove leakage. Test windows are
contiguous and do not overlap, and cost carries across folds; a gap in the shared
OOS dates raises, so missing positions are never silently bridged. The result has
the training summary, per-fold test diagnostics, dated OOS returns and costs, the
verdict recomputed on the stitched series, and input and code digests. An
untouched hold-out and a retrospective OOS are different evidence, and the report
labels the second.

Each factor's Sharpe correction enters the verdict on its own, so a weak
candidate cannot borrow the winner's pass. `FactorWorkflowConfig(trials=...)`
records the cumulative trial count, at least the current number of candidates;
the Sharpe correction uses it, and BH counts omitted candidates conservatively at
p = 1. White and SPA still cover only candidates with supplied returns. By
default only the current list counts, and the report does not claim a complete
search history.

## Conditional diagnostics

`cortexbacktest.pipelines.factor_conditions.diagnose_factor_condition(study, states, name=...)`
takes a walk-forward result and daily states (`date/value/known_at/decision_at`).
Time columns must be time-zone aware; the caller supplies the real decision cutoff
and the matching historical vintage. The function does not audit source timing.

The default threshold is each fold's 0.5 quantile of the available training
states (Polars' default nearest interpolation, so an even sample does not average
the middle two); a fixed threshold can be passed instead. Strictly above the
threshold is high, at or below is low. Too few training states only disables the
estimated threshold. Missing, non-finite or late states are unknown. The output
keeps the original OOS returns and costs and does not recompute positions. Regime
segments count across folds; unknown states or too few segments give
`insufficient_evidence`, otherwise `descriptive_only`. A segment count does not
prove independence.

`mean_spread` / `mean_net_spread` are per-period means and `total_cost` is the
group's summed cost, none annualized; do not compare them with the main report's
annual figures or annualize by state frequency.

## SEC filing ratios

`cortexbacktest.adapters.sec_factors` provides `build_sec_ratio_snapshot` and
`join_sec_ratio_panel`. The first ratios are net margin, operating margin,
liabilities/assets and cash/assets.

```python
from cortexbacktest.adapters.observation_file import load_observation_file
from cortexbacktest.adapters.sec_factors import build_sec_ratio_snapshot, join_sec_ratio_panel

# receipts: one export receipt per concept file at this cutoff, with path and external sha256.
documents = [load_observation_file(r["path"], sha256=r["sha256"], as_of=cutoff)
             for r in receipts]
snapshot = build_sec_ratio_snapshot(
    documents, as_of=cutoff, max_age_days=180, duration="quarterly")
# decisions: date, instrument_id (string), as_of (UTC aware), forward_return.
panel = join_sec_ratio_panel(decisions, [snapshot])
```

A ratio pairs only records with the same CIK, filing version, period and USD
US-GAAP basis. Quarterly or annual must be chosen explicitly; there is no implicit
YTD difference or TTM. The output keeps numerator and denominator record IDs,
periods and availability times, and reports exclusion reasons. The join is exact on
`instrument_id + as_of`; a missing older point stays null and is never back-filled
from the latest snapshot.

## Frozen decisions without future labels

`research.frozen_factor_targets` computes a restored mining selection's targets
as soon as a new decision is available. Unlike `evaluate_holdout`, it needs no
future return labels. It shares frozen input validation, factor builders and
`top_bottom_weights` with the research path; it neither reselects factors nor
changes their parameters. Each selected factor returns its own Polars weight
matrix using the research default top/bottom 20% policy, not an implicit blend.

```python
from cortexbacktest.pipelines.mining_store import load_mining_result
from cortexbacktest.research import frozen_factor_targets

selection = load_mining_result(checkpoint, manifest_sha256=retained_digest)
targets = frozen_factor_targets(
    selection, current_revision_panel,
    inner_execution=frozen_training_execution_panel,
)
```

Pass `inner_execution` only when training bound executable-price evidence; its
fingerprint must match exactly. No execution prices after training are required.
For revision panels, returned `date` values are actual `decision_time` values,
including publication and receipt delays; for plain price matrices they are the
supplied dates. These are decision targets, not fills or executable orders.
Broker sizing, account and cash checks, approval and the handoff to live
execution belong to the downstream integration.

`pipelines.run_frozen_factor_simulation` loads that checkpoint and sends the same
decision targets through `TargetWeightStrategy` and the event simulator. Supply the
research dataset, its matching revision panel, raw OHLCV, and
`BrokerExecutionConfig(opening_times=...)` with explicit executable opening
instants. The simulation starts in cash and returns one `BacktestResult` per
selected factor, with orders, fills, positions and equity. Its manifest binds the
selection digest, revision digest, raw bars and selected factor.

The strategy's `decision_times=True` mode keeps intraday timestamps and uses only
a target strictly older than the opening print. Session-lag research execution is
unchanged. The new mode cannot be combined with a session lag or close execution.
