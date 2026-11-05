"""A rolling short-put study on real SPY chains, end to end.

This is a *mechanical study on one dataset*, not a replication of any paper and
not evidence that the strategy works. Every assumption it makes is listed at the
end of the run and in `docs/option-chain-data-sources.md`; the ones that matter
most are that an end-of-day chain publishes no displayed size (so the run
assumes enough liquidity), and that expiry settles at the vendor's own close
rather than the official OCC fixing.

    python scripts/ingest_kaggle_spy.sh <kaggle dir> data/options
    python examples/spy_short_put_study.py --root data/options \
        --start 2010-01-01 --end 2023-12-31 --output /tmp/spy-short-put
    python examples/spy_short_put_study.py --root data/options --grid \
        --start 2010-01-01 --end 2023-12-31 --output /tmp/spy-short-put

`--grid` sweeps DTE x delta and then prices the search itself: CSCV probability
of backtest overfitting, White's reality check, Hansen's SPA and a Harvey-Liu
haircut on the best Sharpe. Reporting a grid's winner without those numbers is
the thing this repository's statistics exist to prevent.

What it does, on each entry date: sell one put nearest to `--dte` days to expiry
and `--delta` delta, hold it to expiry, and enter again on the next session
after that expiry. Assigned stock is sold at the next frame, so the result stays
an option result. Selection sees only the row's own snapshot.

The two passes matter. Selection runs entirely in Polars over the whole store,
and only the contracts actually traded are turned into `OptionMarketFrame`
objects — profiled, that adapter is one to two orders of magnitude slower than any DataFrame stage, so
materialising fourteen years of chains as objects would cost tens of gigabytes
to no purpose.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

import numpy as np
import polars as pl

from cortexbacktest.options.chains import CHAIN_KEY, CHAIN_SCHEMA
from cortexbacktest.options import (
    ChainTimePolicy, OptionBacktestConfig, ResearchOptionCosts, OptionChainStore, QuotePolicy, RegTOptionMargin,
    ResearchOptionMargin, attach_lifecycle, expiry_events_from_prices,
    expiry_events_from_underlying,
    option_chain_to_frames, option_contracts_from_chain, run_option_backtest,
    run_option_scenarios,
)
from cortexbacktest.options import probability_of_backtest_overfitting
from cortexbacktest.event.financing import RateSchedule
from cortexbacktest.research import (
    reality_check, sharpe_haircut, superior_predictive_ability,
)

#: What holds for every run. The two that depend on `--closes` and `--rates`
#: are decided per run by `assumptions_for`, because a list that says "no
#: financing configured" under a run financed from a published curve is worse
#: than no list at all.
ASSUMPTIONS = [
    "EOD chains publish no displayed size: run assumes sufficient liquidity",
    "16:00 America/New_York snapshot; PM settlement assumed",
    "no dividends and no stock borrow modelled",
    "one dataset, one underlying, no out-of-sample holdout",
]


def assumptions_for(args) -> list[str]:
    """`ASSUMPTIONS` plus what this particular run did about fixings and cash."""
    fixing = (f"expiry settles at the close supplied in {Path(args.closes).name}, "
              "an independent source but still not the official OCC fixing"
              if getattr(args, "closes", None) else
              "expiry settles at the vendor's own close, not the official OCC fixing")
    curve = financing_for(getattr(args, "rates", None))
    if curve is None:
        return [*ASSUMPTIONS, fixing,
                "cash financed at 0%: for a cash-secured programme this understates the result"]
    # Financing raises the Sharpe and the P&L without the strategy doing
    # anything, so saying where it came from is not enough -- over 2010-2023 the
    # DGS1MO curve took this study from Sharpe 0.0235 (p=0.18) to 0.0475
    # (p=0.0061) on identical option fills. The bill is not skill.
    return [*ASSUMPTIONS, fixing,
            f"cash financed from {curve.source}, debit +150bp; borrow 0",
            "returns are TOTAL, not excess: the financing leg is the T-bill and belongs in "
            "the benchmark too, so the Sharpe and p-value here credit the strategy with the "
            "risk-free rate and are not comparable to a 0%-financed run"]

EASTERN_CLOSE = ChainTimePolicy(quote_time=time(16, 0), expiry_time=time(16, 0),
                                zone="America/New_York")


def candidates(store: OptionChainStore, *, underlying: str, start: datetime, end: datetime,
               dte: float, delta: float, dte_window: float, spread_cap: float) -> pl.DataFrame:
    """One candidate row per session: the put closest to the target DTE and delta.

    Filtering and ranking are expressions over the store, so this reads the
    whole sample without materialising it. Every column used is from the row's
    own snapshot, so nothing here can see past the session it ranks.
    """
    ranked = (
        store.scan(underlyings=[underlying], start=start, end=end)
        .filter(
            (pl.col("right") == "PUT")
            & pl.col("dte").is_between(dte - dte_window, dte + dte_window)
            & (pl.col("bid") > 0)
            & pl.col("delta").is_not_null()
            & pl.col("delta").is_between(-0.95, -0.05)
            & pl.col("underlying_price").is_not_null()
            & ((pl.col("ask") - pl.col("bid")) <= spread_cap * pl.col("mid"))
        )
        .with_columns(
            (pl.col("dte") - dte).abs().alias("_dte_distance"),
            (pl.col("delta") - delta).abs().alias("_delta_distance"),
        )
        .sort(["quote_date", "_dte_distance", "_delta_distance", "strike", "option_symbol"])
        .group_by("quote_date", maintain_order=True)
        .first()
    )
    return ranked.select("quote_date", "option_symbol", "expiry", "strike", "delta",
                         "bid", "ask", "mid", "implied_volatility", "underlying_price").collect()


def store_sessions(store: OptionChainStore, *, underlying: str) -> set:
    """Every session the store actually has a row for."""
    return set(store.scan(underlyings=[underlying])
               .select(pl.col("quote_date").unique()).collect().to_series().to_list())


def schedule(picks: pl.DataFrame, sessions: set) -> tuple[list[dict], int]:
    """Non-overlapping rolls: enter, hold to expiry, re-enter on the next session.

    A candidate whose expiry date is missing from the data is skipped rather
    than settled somewhere else. The Kaggle SPY dataset drops about four
    sessions a year, and 49 of its 1,572 expiry dates have no row at all; those
    positions could only be closed at a price from a different day, which is a
    fabricated fixing however it is dressed up.
    """
    entries: list[dict] = []
    available = None
    unsettleable = 0
    for row in picks.iter_rows(named=True):
        if available is not None and row["quote_date"] <= available:
            continue
        if row["expiry"] not in sessions:
            unsettleable += 1
            continue
        entries.append(row)
        available = row["expiry"]
    return entries, unsettleable


def traded_chain(store: OptionChainStore, entries: list[dict], *, underlying: str) -> pl.DataFrame:
    """Every quote of every contract actually traded, from entry to expiry.

    The engine refuses to value a held position without a fresh quote, so a
    contract that stops being quoted mid-life must fail loudly rather than be
    marked at something invented. Nothing is filled in here.
    """
    symbols = [entry["option_symbol"] for entry in entries]
    first, last = entries[0]["quote_date"], max(entry["expiry"] for entry in entries)
    return (store.scan(underlyings=[underlying], start=first, end=last)
            .filter(pl.col("option_symbol").is_in(symbols))
            .collect())


def session_anchors(store: OptionChainStore, expiries: set, *, underlying: str) -> pl.DataFrame:
    """One real row per expiry session, so the run has a frame to settle in.

    A contract is not always quoted on its own expiry date — 30 of the 2019
    entries were not — and the engine refuses to settle a position without a
    frame carrying the fixing. Rather than fabricate one, this pulls a real row
    from that session (the near-the-money put, chosen deterministically) purely
    so the session exists as a frame with the underlying's close on it. The
    anchor contract is never traded.
    """
    if not expiries:
        return pl.DataFrame(schema=dict(CHAIN_SCHEMA))
    window = sorted(expiries)
    return (store.scan(underlyings=[underlying], start=window[0], end=window[-1])
            .filter(pl.col("quote_date").is_in(window)
                    & (pl.col("right") == "PUT")
                    & pl.col("underlying_price").is_not_null())
            .with_columns((pl.col("strike") - pl.col("underlying_price")).abs().alias("_atm"))
            .sort(["quote_date", "_atm", "dte", "option_symbol"])
            .group_by("quote_date", maintain_order=True)
            .first()
            .drop("_atm")
            .collect())


def quote_gaps(chain: pl.DataFrame, entries: list[dict], max_age: timedelta) -> set[str]:
    """Contracts whose quotes stop for longer than the run's staleness tolerance.

    A shorter gap is fine: the engine values a held position from its last quote
    while that quote is inside `QuotePolicy.max_age`, which is what a real book
    does over a weekend. A longer one means the position would be marked at a
    price nobody published, so the entry is dropped and counted.
    """
    stale: set[str] = set()
    for entry in entries:
        quoted = sorted(chain.filter(pl.col("option_symbol") == entry["option_symbol"])
                        ["quote_date"].to_list())
        if not quoted or quoted[0] > entry["quote_date"]:
            stale.add(entry["option_symbol"])
            continue
        spans = [b - a for a, b in zip(quoted, quoted[1:])]
        if (spans and max(spans) > max_age) or entry["expiry"] - quoted[-1] > max_age:
            stale.add(entry["option_symbol"])
    return stale


def plan(store: OptionChainStore, args, *, dte: float, delta: float,
         sessions: set) -> tuple[list[dict], dict]:
    """Entries for one configuration, with the reasons anything was dropped."""
    start = datetime.combine(args.start, time(0), timezone.utc)
    end = datetime.combine(args.end, time(23, 59), timezone.utc)
    picks = candidates(store, underlying=args.underlying, start=start, end=end, dte=dte,
                       delta=delta, dte_window=args.dte_window, spread_cap=args.spread_cap)
    if picks.is_empty():
        return [], {"candidates_skipped_no_expiry_session": 0,
                    "entries_dropped_for_quote_gaps": 0}
    entries, unsettleable = schedule(picks, sessions)
    max_age = timedelta(days=args.max_quote_age)
    stale: set[str] = set()
    if entries:
        stale = quote_gaps(traded_chain(store, entries, underlying=args.underlying),
                           entries, max_age)
        entries = [entry for entry in entries if entry["option_symbol"] not in stale]
    return entries, {"candidates_skipped_no_expiry_session": unsettleable,
                     "entries_dropped_for_quote_gaps": len(stale)}


def market(store: OptionChainStore, args, plans: list[list[dict]]) -> tuple[tuple, tuple, int]:
    """One frame set covering every configuration in the run.

    Every strategy has to be marked on the same instants or the T x N matrix that
    CSCV, the reality check and SPA all require is not synchronised — comparing
    strategies observed on different days would compare their calendars as much
    as their signals. So the traded contracts of the whole grid are assembled
    once, and each configuration is run over the same frames while holding
    whatever it chose.
    """
    entries = [entry for one in plans for entry in one]
    chain = traded_chain(store, entries, underlying=args.underlying)
    covered = set(chain["quote_date"].to_list())
    anchors = session_anchors(store, {entry["expiry"] for entry in entries} - covered,
                              underlying=args.underlying)
    if not anchors.is_empty():
        chain = pl.concat([chain, anchors]).unique(
            subset=["quote_date", "option_symbol"], keep="first").sort(list(CHAIN_KEY))
    contracts = option_contracts_from_chain(chain)
    closes = session_close_table(getattr(args, "closes", None))
    events = (expiry_events_from_prices(closes, contracts, time_policy=EASTERN_CLOSE)
              if closes is not None else
              expiry_events_from_underlying(chain, contracts, time_policy=EASTERN_CLOSE))
    frames = attach_lifecycle(
        option_chain_to_frames(chain, time_policy=EASTERN_CLOSE), events)
    return contracts, frames, anchors.height


def session_close_table(path) -> dict[tuple[str, date], float] | None:
    """`ticker,session,close` -> the fixings, or None to settle off the chain.

    A chain's own `underlying_price` has one vendor marking both sides of the
    trade. Supplying an independently sourced close does not make the fixing
    official -- an equity option settles against the OCC's determination, not
    against anyone's close -- but it does let the two disagree.
    """
    if path is None:
        return None
    table = pl.read_csv(path, try_parse_dates=True)
    missing = {"ticker", "session", "close"} - set(table.columns)
    if missing:
        raise SystemExit(f"--closes needs columns ticker,session,close (missing {sorted(missing)})")
    return {(row["ticker"], row["session"]): float(row["close"])
            for row in table.iter_rows(named=True)}


def financing_for(path) -> RateSchedule | None:
    """A FRED `observation_date,VALUE` CSV as a dated financing curve.

    Without one, a run finances at 0%, which for a cash-secured programme is a
    silent understatement: over 2010-2023 the DGS1MO curve is worth 14,357 on
    100,000 of cash, more than the option P&L of the default parameters. `.` and
    an empty field both mark a non-publication day and are stepped over.
    """
    if path is None:
        return None
    rows = []
    for line in Path(path).read_text().splitlines()[1:]:
        day, _, value = line.partition(",")
        if value.strip() not in (".", ""):
            rows.append((date.fromisoformat(day.strip()), float(value)))
    if not rows:
        raise SystemExit(f"{path} carries no observations")
    # 150bp over the bill on a debit balance is a broker's typical markup and is
    # a choice, not a published number.
    return RateSchedule.from_annual_percent(rows, source=f"FRED {Path(path).stem}",
                                            debit_spread=0.015)


def strategy_for(entries: list[dict], args):
    targets = {entry["quote_date"]: {entry["option_symbol"]: -args.contracts}
               for entry in entries}

    def strategy(frame, account):
        orders = dict(targets.get(frame.timestamp, {}))
        # An assigned put delivers stock. A rolling short-put programme sells it
        # at the next opportunity instead of quietly turning into a long-equity
        # position whose P&L would swamp the option result.
        if account.positions.get(args.underlying, 0):
            orders[args.underlying] = 0.0
        return orders

    return strategy


def configuration(args) -> OptionBacktestConfig:
    return OptionBacktestConfig(
        initial_cash=args.cash, require_sizes=False,
        costs=ResearchOptionCosts(
            spread_fraction=1.0, option_commission=args.fee,
            option_slippage_ticks=args.slippage_ticks,
            impact_coefficient=args.impact, impact_reference=args.impact_reference),
        margin=RegTOptionMargin() if args.margin == "reg-t" else ResearchOptionMargin(),
        quote_policy=QuotePolicy(max_age=timedelta(days=args.max_quote_age),
                                 allow_zero_bid=True),
        financing=financing_for(getattr(args, "rates", None)),
        data_id=f"kaggle/spy-eod/{args.start}..{args.end}", strategy_id="short-put/1")


def run(args) -> dict:
    store = OptionChainStore(args.root)
    sessions = store_sessions(store, underlying=args.underlying)
    entries, dropped = plan(store, args, dte=args.dte, delta=args.delta, sessions=sessions)
    if not entries:
        raise SystemExit("no candidate could be settled inside this window")
    contracts, frames, anchored = market(store, args, [entries])
    strategy = strategy_for(entries, args)
    config = configuration(args)
    scenarios = run_option_scenarios(contracts, frames, lambda: strategy,
                                     config=config, spread_fractions=(0.0, 0.5, 1.0))
    full = scenarios[1.0]
    equity = full.snapshots["equity"].to_numpy()
    returns = equity[1:] / equity[:-1] - 1
    sharpe = float(returns.mean() / returns.std(ddof=1)) if returns.std(ddof=1) > 0 else 0.0
    haircut = (sharpe_haircut(sharpe, observations=len(returns), trials=args.trials)
               if sharpe > 0 else None)
    return {
        "parameters": {"dte": args.dte, "delta": args.delta, "contracts": args.contracts,
                       "margin": args.margin, "fee": args.fee, "trials_declared": args.trials},
        "sample": {"sessions": len(frames), "entries": len(entries),
                   "first": str(frames[0].timestamp), "last": str(frames[-1].timestamp),
                   "contracts_traded": len(contracts),
                   "entries_dropped_for_quote_gaps": dropped["entries_dropped_for_quote_gaps"],
                   "candidates_skipped_no_expiry_session": dropped["candidates_skipped_no_expiry_session"],
                   "settlement_sessions_anchored": anchored},
        "scenarios": {str(fraction): result.summary()
                      for fraction, result in scenarios.items()},
        "attribution_at_full_spread": dict(full.attribution),
        "per_frame_sharpe": sharpe,
        "haircut": None if haircut is None else {
            "haircut_sharpe": haircut.haircut_sharpe, "haircut": haircut.haircut,
            "p_value": haircut.p_value, "adjusted_p_value": haircut.adjusted_p_value},
        "assumptions": assumptions_for(args),
        "data_digest": full.manifest["data_digest"],
    }


def label(column: dict) -> str:
    return f"dte={column['dte']:g},delta={column['delta']:g}"


def grid(args) -> dict:
    """Run every (DTE, delta) pair over one shared frame set, then price the search.

    A grid is where option research goes wrong: nine configurations produce a
    best Sharpe whether or not any of them has an edge, and reporting that one
    on its own is the data snooping the statistics below exist to price. So the
    matrix of per-frame returns goes through CSCV (how often the in-sample best
    lands in the bottom half out of sample), White's reality check and Hansen's
    SPA (is the best better than doing nothing, given that nine were tried), and
    the reported Sharpe is haircut for the size of the search.
    """
    store = OptionChainStore(args.root)
    sessions = store_sessions(store, underlying=args.underlying)
    points = [(dte, delta) for dte in args.dte_grid for delta in args.delta_grid]
    plans, drops = [], []
    for dte, delta in points:
        entries, dropped = plan(store, args, dte=dte, delta=delta, sessions=sessions)
        plans.append(entries)
        drops.append(dropped)
    if not any(plans):
        raise SystemExit("no configuration produced a settleable entry")
    contracts, frames, anchored = market(store, args, plans)
    config = configuration(args)

    columns, results = [], []
    for (dte, delta), entries in zip(points, plans):
        if not entries:
            continue
        result = run_option_backtest(contracts, frames, strategy_for(entries, args),
                                     config=replace(config, strategy_id=f"short-put/{dte}/{delta}"))
        equity = result.snapshots["equity"].to_numpy()
        columns.append({"dte": dte, "delta": delta, "entries": len(entries),
                        "summary": result.summary()})
        results.append(equity)

    matrix = np.column_stack(results)
    returns = matrix[1:] / matrix[:-1] - 1
    sharpes = returns.mean(axis=0) / returns.std(axis=0, ddof=1)
    best = int(np.argmax(sharpes))
    for column, sharpe in zip(columns, sharpes):
        column["per_frame_sharpe"] = float(sharpe)

    # CSCV needs an even block count that divides the sample, so the tail is
    # trimmed rather than the blocks made ragged; the discarded rows are named.
    blocks = args.blocks
    usable = len(returns) - len(returns) % blocks
    pbo = probability_of_backtest_overfitting(returns[:usable], blocks=blocks)
    check = reality_check(returns, resamples=args.resamples, block_size=args.block_size, seed=7)
    spa = superior_predictive_ability(returns, resamples=args.resamples,
                                      block_size=args.block_size, seed=7)
    haircut = (sharpe_haircut(float(sharpes[best]), observations=len(returns),
                              trials=len(columns))
               if sharpes[best] > 0 else None)
    return {
        "parameters": {"dte_grid": args.dte_grid, "delta_grid": args.delta_grid,
                       "contracts": args.contracts, "margin": args.margin, "fee": args.fee,
                       "slippage_ticks": args.slippage_ticks, "impact": args.impact},
        "sample": {"sessions": len(frames), "configurations": len(columns),
                   "contracts_traded": len(contracts), "settlement_sessions_anchored": anchored,
                   "first": str(frames[0].timestamp), "last": str(frames[-1].timestamp),
                   "dropped_per_configuration": drops,
                   "cscv_rows_trimmed": len(returns) - usable},
        "configurations": columns,
        "best": {"dte": columns[best]["dte"], "delta": columns[best]["delta"],
                 "per_frame_sharpe": float(sharpes[best])},
        "statistics": {
            "cscv_pbo": pbo["pbo"], "cscv_combinations": pbo["combinations"],
            # The three pick their own winner: the reality check by mean, SPA by
            # studentised mean, the haircut by the Sharpe a researcher would
            # have reported. They need not agree, and when they do not, that is
            # itself worth seeing.
            "reality_check_p": check.p_value,
            "reality_check_best": label(columns[check.best]),
            "spa_p": spa.p_value, "spa_best": label(columns[spa.best]),
            "spa_p_lower": spa.p_lower, "spa_p_upper": spa.p_upper,
            "haircut": None if haircut is None else {
                "haircut_sharpe": haircut.haircut_sharpe, "haircut": haircut.haircut,
                "p_value": haircut.p_value, "adjusted_p_value": haircut.adjusted_p_value},
        },
        "assumptions": assumptions_for(args) + [
            "all configurations share one frame set, so the return matrix is synchronised",
            f"trials declared to the haircut = {len(columns)}, the grid size only: any "
            "configuration tried before this run and not reported belongs in that count",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path("data/options"))
    parser.add_argument("--underlying", default="SPY")
    parser.add_argument("--start", type=date.fromisoformat, default=date(2010, 1, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2023, 12, 31))
    parser.add_argument("--dte", type=float, default=30)
    parser.add_argument("--dte-window", type=float, default=10)
    parser.add_argument("--delta", type=float, default=-0.30)
    parser.add_argument("--spread-cap", type=float, default=0.5)
    parser.add_argument("--contracts", type=int, default=1)
    parser.add_argument("--cash", type=float, default=100_000)
    parser.add_argument("--fee", type=float, default=0.65,
                        help="commission per contract")
    parser.add_argument("--slippage-ticks", type=float, default=0.0,
                        help="ticks paid beyond the touch; an EOD chain cannot measure "
                             "this, so it is a declared assumption")
    parser.add_argument("--impact", type=float, default=0.0,
                        help="market impact in half-spreads at --impact-reference contracts")
    parser.add_argument("--impact-reference", type=float, default=100.0)
    parser.add_argument("--margin", choices=["reg-t", "unnetted"], default="reg-t")
    parser.add_argument("--rates", type=Path,
                        help="FRED CSV (e.g. DGS1MO) used as the financing curve; "
                             "without it the run finances at 0%%")
    parser.add_argument("--closes", type=Path,
                        help="ticker,session,close table used for expiry fixings "
                             "instead of the chain's own underlying price")
    parser.add_argument("--max-quote-age", type=int, default=5,
                        help="days a held position may be valued from its last quote")
    parser.add_argument("--trials", type=int, default=1,
                        help="configurations searched before reporting this one; the "
                             "Sharpe haircut is meaningless if this is understated")
    parser.add_argument("--grid", action="store_true",
                        help="sweep --dte-grid x --delta-grid and price the search")
    parser.add_argument("--dte-grid", type=lambda v: [float(x) for x in v.split(",")],
                        default=[21.0, 30.0, 45.0])
    parser.add_argument("--delta-grid", type=lambda v: [float(x) for x in v.split(",")],
                        default=[-0.20, -0.30, -0.40])
    parser.add_argument("--blocks", type=int, default=8, help="CSCV blocks (even, 2..12)")
    parser.add_argument("--resamples", type=int, default=1000)
    parser.add_argument("--block-size", type=float, default=10.0,
                        help="mean stationary-bootstrap block length, in frames")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = grid(args) if args.grid else run(args)
    text = json.dumps(report, indent=2, default=str)
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        name = "short-put-grid.json" if args.grid else "short-put-study.json"
        (args.output / name).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
