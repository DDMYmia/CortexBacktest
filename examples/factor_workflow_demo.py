"""The whole pass: a point-in-time panel, a factor family, and a verdict.

    scripts/build_factor_panel.py data/panel --start 2015 --end 2026
    examples/factor_workflow_demo.py --panel data/panel --output /tmp/factors

This is the reference shape of a factor study in this repository. It is not a
strategy and none of these eight factors is a recommendation; what it fixes is
the *order* of the steps and the things that cannot be skipped between them.

Three choices here decide more than the factor definitions do:

**Open-to-open, not close-to-close.** A factor computed from today's close
cannot be traded at today's close. Scoring it against close-to-close returns
credits it with a move it could never have captured, and that alone is usually
the difference between a factor that looks tradable and one that is.

**A liquidity screen on the signal date only.** Filtering on the return date
selects on the future; the screen has to be a decision the trader could have
made in advance.

**The family is priced, not the winner.** Eight candidates are eight draws.
`run_factor_workflow` computes the family-level corrections over all of them and
will not hand back an evaluation without them.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

from cortexbacktest.pipelines.factor_workflow import (
    FactorWorkflowConfig, run_factor_workflow,
)
from cortexbacktest.pipelines.factor_walk_forward import (
    FactorWalkForwardConfig, run_factor_walk_forward,
)

#: A quantity known at the close of its own row's date. Order matters only for
#: readability; each is independent.
FACTORS = (
    "momentum_12_1", "reversal_5d", "low_volatility", "illiquidity",
    "close_volume_tilt", "smoothness", "overnight_drift", "close_over_vwap",
)


def features(panel: pl.LazyFrame) -> pl.LazyFrame:
    over = {"order_by": "date"}
    sessions = panel.select("date").unique().sort("date").with_row_index("_session")
    panel = panel.join(sessions, on="date")
    panel = panel.with_columns(
        (pl.col("close").log() - pl.col("close").log().shift(1).over("symbol", **over)).alias("ret"),
        (pl.col("open").log() - pl.col("close").log().shift(1).over("symbol", **over)).alias("overnight"),
        (pl.col("realized_variance") * 252).sqrt().alias("rv_annual"),
    )
    return panel.with_columns(
        # 12-1 momentum: the classic, skipping the most recent month.
        (pl.col("close").log().shift(21).over("symbol", **over)
         - pl.col("close").log().shift(252).over("symbol", **over)).alias("momentum_12_1"),
        (-pl.col("ret").rolling_sum(5).over("symbol", **over)).alias("reversal_5d"),
        (-pl.col("rv_annual").rolling_mean(21).over("symbol", **over)).alias("low_volatility"),
        # Amihud: absolute return per dollar traded, scaled to stay readable.
        ((pl.col("ret").abs() / pl.col("dollar_volume")).rolling_mean(21).over("symbol", **over)
         * 1e9).alias("illiquidity"),
        # The four below need the minute tape; a daily file cannot produce them.
        pl.col("close_volume_share").rolling_mean(21).over("symbol", **over).alias("close_volume_tilt"),
        (-(pl.col("max_minute_move") / pl.col("rv_per_minute")).rolling_mean(21)
         .over("symbol", **over)).alias("smoothness"),
        pl.col("overnight").rolling_mean(21).over("symbol", **over).alias("overnight_drift"),
        ((pl.col("close") - pl.col("vwap")) / pl.col("vwap")).rolling_mean(21)
        .over("symbol", **over).alias("close_over_vwap"),
        # Entered at the first open after the signal, exited one open later.
        pl.when((pl.col("_session").shift(-1).over("symbol", **over) == pl.col("_session") + 1)
                & (pl.col("_session").shift(-2).over("symbol", **over) == pl.col("_session") + 2))
        .then(pl.col("open").log().shift(-2).over("symbol", **over)
              - pl.col("open").log().shift(-1).over("symbol", **over)).alias("forward_return"),
    )


def forward_returns(panel: pl.LazyFrame, quotes: pl.LazyFrame | None) -> pl.LazyFrame:
    """log open[t+2] - log open[t+1] over the global session calendar.

    Opens come from the panel and, where the panel has no row for a name on
    those sessions (it left that month's universe), from `quotes` fetched with
    the same definition (`scripts/fill_holding_quotes.py`). A label whose
    opens exist in neither stays missing, so the workflow still refuses it.
    """
    sessions = panel.select(pl.col("date").unique().sort()).with_row_index("_s")
    opens = panel.select("symbol", "date", "open")
    if quotes is not None:
        extra = quotes.rename({"ticker": "symbol", "day": "date"}).select("symbol", "date", "open")
        opens = pl.concat([opens, extra.join(opens, on=["symbol", "date"], how="anti")])
    opens = opens.join(sessions, on="date").select("symbol", "_s", "open")
    later = lambda k: opens.select("symbol", (pl.col("_s") - k).alias("_s"), pl.col("open").alias(f"_o{k}"))  # noqa: E731
    return (panel.select("symbol", "date").join(sessions, on="date")
            .join(later(1), on=["symbol", "_s"], how="left")
            .join(later(2), on=["symbol", "_s"], how="left")
            .select("symbol", "date", (pl.col("_o2").log() - pl.col("_o1").log()).alias("forward_return")))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--panel", type=Path, default=Path("data/panel"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--walk-forward", action="store_true")
    parser.add_argument("--train-sessions", type=int, default=756)
    parser.add_argument("--test-sessions", type=int, default=252)
    parser.add_argument("--embargo-sessions", type=int, default=0)
    parser.add_argument("--min-dollar-volume", type=float, default=5e6)
    parser.add_argument("--holding-quotes", type=Path,
                        help="opens for names outside a month's universe (scripts/fill_holding_quotes.py)")
    parser.add_argument("--unresolved-labels", choices=("refuse", "exclude"), default="refuse",
                        help="exclude drops rows whose forward opens exist nowhere and counts them in the manifest")
    parser.add_argument("--cost-bps", type=float, default=10.0,
                        help="one-way cost asserted against realised turnover")
    args = parser.parse_args()

    panel = (pl.scan_parquet(args.panel)
             .rename({"ticker": "symbol", "day": "date"})
             .sort("symbol", "date"))
    quotes = pl.scan_parquet(args.holding_quotes) if args.holding_quotes else None
    frame = (features(panel)
             .drop("forward_return")
             .join(forward_returns(panel, quotes), on=["symbol", "date"], how="left")
             # Only the final two global sessions are right-censored by the
             # requested extraction endpoint. Interior label gaps must fail.
             .filter(pl.col("_session") < pl.col("_session").max() - 1)
             .filter(pl.col("dollar_volume") > args.min_dollar_volume)
             .select("date", "symbol", "forward_return", *FACTORS)
             .collect())

    settings = FactorWorkflowConfig(cost_bps=args.cost_bps, unresolved_labels=args.unresolved_labels)
    if args.walk_forward:
        result = run_factor_walk_forward(
            frame, FACTORS, config=settings, data_id=str(args.panel),
            split=FactorWalkForwardConfig(args.train_sessions, args.test_sessions, 2,
                                          embargo_sessions=args.embargo_sessions),
            label_definition="log open[t+2] - log open[t+1]; signal at close[t]",
        )
    else:
        result = run_factor_workflow(frame, FACTORS, config=settings, data_id=str(args.panel))
    print(result.summary())

    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "factor-report.json").write_text(
            json.dumps(result.report(), indent=2, default=str))
        print(f"\nwrote {args.output / 'factor-report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
