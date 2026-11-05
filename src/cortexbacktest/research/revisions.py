"""Build causal factors from received price revisions, without flattening history."""
from __future__ import annotations

from typing import TYPE_CHECKING, Literal
from dataclasses import dataclass, replace
from datetime import datetime
from hashlib import sha256

import polars as pl

from .factor_expression import BUILDERS


if TYPE_CHECKING:
    from .mining import MiningFactorSpec


@dataclass(frozen=True)
class RevisionPanel:
    """One fixed study universe with explicit session and decision clocks.

    ``head`` reserves later decisions without discarding earlier revisions.
    Fingerprints cover only evidence available at the last decision, so new
    later receipts cannot rewrite the frozen inner study.
    Explicit unverified_price_basis invalidations carry null prices. Selection
    precedes usability: an invalid latest receipt masks the previous price.
    ``membership`` (symbol, start, end as session-close timestamps, inclusive)
    makes the universe point in time: a factor value exists only on decisions
    where its symbol was a member. Prices outside membership still warm up
    lookbacks and settle labels; they are never ranked or held.
    ``rebalance`` = "weekly" holds each factor value from the first decision of
    its UTC calendar week (Monday) through the week, so targets change at most
    once a week; the value is the one that first decision computed, in research
    and in a live decision alike.
    """
    revisions: pl.DataFrame
    decisions: pl.DataFrame
    knowledge_policy: Literal["received", "source_publication_replay"] = "received"
    replay_as_of: datetime | None = None
    membership: pl.DataFrame | None = None
    rebalance: Literal["daily", "weekly"] = "daily"

    def __post_init__(self):
        if self.rebalance not in ("daily", "weekly"):
            raise ValueError("Unsupported rebalance frequency")
        if self.replay_as_of is not None and (self.knowledge_policy != "source_publication_replay"
                or self.replay_as_of.utcoffset() is None):
            raise ValueError("Forward replay requires an aware snapshot cutoff")
        if self.knowledge_policy not in ("received", "source_publication_replay"):
            raise ValueError("Unsupported research knowledge policy")
        revisions, decisions = _prepare(self.revisions, self.decisions)
        object.__setattr__(self, "revisions", revisions)
        object.__setattr__(self, "decisions", decisions)
        if self.membership is not None:
            object.__setattr__(self, "membership", _membership(self.membership))

    def head(self, size: int) -> RevisionPanel:
        return replace(self, decisions=self.decisions.head(size))

    def through(self, end) -> RevisionPanel:
        return replace(self, decisions=self.decisions.filter(pl.col("date") <= end))

    def _week_start(self, index: int) -> int:
        """The first decision of decision `index`'s UTC week (itself when rebalancing daily)."""
        if self.rebalance == "daily":
            return index
        weeks = self.decisions["date"].dt.truncate("1w")
        week = weeks[index]
        return int(weeks.slice(0, index + 1).eq(week).arg_true()[0])

    def held(self, frame: pl.DataFrame) -> pl.DataFrame:
        """A date x symbol factor frame held from each week's first decision (unchanged when daily)."""
        if self.rebalance == "daily":
            return frame
        return frame.sort("date").with_columns(pl.exclude("date").first().over(pl.col("date").dt.truncate("1w")))

    def members(self, frame: pl.DataFrame) -> pl.DataFrame:
        """A date x symbol frame with every cell outside membership set to null."""
        if self.membership is None:
            return frame
        spans: dict[str, pl.Expr] = {}
        for row in self.membership.iter_rows(named=True):
            inside = (pl.col("date") >= row["start"]) & (pl.col("date") <= row["end"])
            spans[row["symbol"]] = spans[row["symbol"]] | inside if row["symbol"] in spans else inside
        return frame.with_columns(
            pl.when(spans.get(symbol, pl.lit(False))).then(pl.col(symbol)).otherwise(None).alias(symbol)
            for symbol in frame.columns if symbol != "date")

    def visible_revisions(self, cutoff: datetime) -> pl.DataFrame:
        """Replay uses exported source versions; original receipt clocks remain evidence."""
        visible = self.revisions.filter(pl.col("known_at") <= cutoff)
        if self.knowledge_policy == "received":
            visible = visible.filter(pl.col("recorded_at") <= cutoff)
        elif self.replay_as_of is not None:
            visible = visible.filter((pl.col("recorded_at") <= self.replay_as_of)
                                     | (pl.col("recorded_at") <= cutoff))
        return visible

    def prices(self, *, cutoff: datetime | None = None, grid: pl.DataFrame | None = None) -> pl.DataFrame:
        decision_cutoff = cutoff if cutoff is not None else self.decisions.item(-1, "decision_time")
        dates = grid if grid is not None else self.decisions.select("date")
        visible = self.visible_revisions(decision_cutoff).filter(
            pl.col("date").is_in(dates["date"].implode()) & (pl.col("date") <= decision_cutoff)
        ).sort("recorded_at").unique(subset=["date", "symbol"], keep="last")
        symbols = sorted(self.revisions["symbol"].unique())
        panel = dates.join(visible.pivot(on="symbol", index="date", values="price"), on="date", how="left")
        return panel.with_columns(pl.lit(None, dtype=pl.Float64).alias(s) for s in symbols if s not in panel.columns).select(
            "date", *symbols).sort("date")

    def fingerprint(self) -> str:
        cutoff = self.decisions[-1, "decision_time"]
        visible = self.visible_revisions(cutoff).filter(
            (pl.col("date") <= self.decisions[-1, "date"])
        ).sort("date", "symbol", "recorded_at")
        digest = sha256(self.decisions.write_csv().encode())
        if self.knowledge_policy != "received":
            digest.update(self.knowledge_policy.encode())
        if self.membership is not None:
            digest.update(b"membership:" + self.membership.write_csv().encode())
        if self.rebalance != "daily":
            digest.update(b"rebalance:" + self.rebalance.encode())
        digest.update(visible.write_csv().encode())
        return digest.hexdigest()

    def factor(self, spec: MiningFactorSpec, *, last_only: bool = False) -> pl.DataFrame:
        # Operators are date-local. Build each leaf using its own receipt-cutoff
        # loop, then combine the full causal columns rather than reranking each
        # overlapping lookback window for every decision.
        if spec.indicator not in BUILDERS:
            # Leaves are masked first, so a cross-sectional rank compares members only.
            return self.members(self.held(spec.compose(lambda component: self.factor(component, last_only=last_only))))
        # Each decision is built on its own window, as it saw it, so a value never depends on how
        # much earlier history is loaded. A decision whose window has not been revised since its
        # cutoff sees exactly the final panel's rows: those are sliced from one pivot instead of
        # re-selecting every revision per decision.
        if last_only and self.rebalance != "daily":
            # The live decision holds what its week's first decision computed, as research did.
            # Membership is judged on the live decision's own date, as research judges each held day.
            last = self.decisions.height - 1
            first = replace(self.through(self.decisions[self._week_start(last), "date"]),
                            rebalance="daily", membership=None)
            row = first.factor(spec, last_only=True)
            return self.members(row.with_columns(pl.lit(self.decisions[last, "date"]).alias("date")))
        indices = [self.decisions.height - 1] if last_only else list(range(self.decisions.height))
        settled = self._settled(spec.window + 1)
        final = self.prices()
        # Settled windows share one query, grouped independently so rolling kernels reset at
        # every decision. Revised windows still use their own receipt-cutoff view.
        windows = [i for i in indices if settled[i]]
        revised = [self._factor_at(spec, i) for i in indices if not settled[i]]
        rows = build_windows(spec, final, windows)
        return self.members(self.held(pl.concat([*rows, *revised]).sort("date") if revised else pl.concat(rows)))

    def _factor_at(self, spec: MiningFactorSpec, index: int) -> pl.DataFrame:
        """The factor exactly as decision `index` saw it: its window of prices at its cutoff."""
        grid = self.decisions.slice(*_window(spec, index)).select("date")
        panel = self.prices(cutoff=self.decisions[index, "decision_time"], grid=grid)
        return spec.build(panel).tail(1)

    def _settled(self, rows: int) -> pl.Series:
        """Per decision: no price in its last `rows` sessions became visible after its cutoff.

        A revision counts from the moment the knowledge policy lets a decision see it; one the
        final decision cannot see is invisible to every decision and changes nothing.
        """
        last = self.decisions[-1, "decision_time"]
        if self.knowledge_policy == "received":
            visible_from = pl.max_horizontal("known_at", "recorded_at")
        elif self.replay_as_of is None:
            visible_from = pl.col("known_at")
        else:
            visible_from = pl.when(pl.col("recorded_at") <= self.replay_as_of).then(pl.col("known_at")).otherwise(
                pl.max_horizontal("known_at", "recorded_at"))
        latest = self.revisions.with_columns(visible_from.alias("visible_from")).filter(
            pl.col("visible_from") <= last).group_by("date").agg(pl.col("visible_from").max())
        grid = self.decisions.join(latest, on="date", how="left").sort("date")
        newest = grid["visible_from"].rolling_max(window_size=rows, min_samples=1)
        return (newest <= grid["decision_time"]).fill_null(True)



WINDOW_BATCH_CELLS = 1_000_000


def _window(spec: MiningFactorSpec, index: int) -> tuple[int, int]:
    """(offset, length) of decision `index`'s window: itself and up to `spec.window` sessions before."""
    start = max(0, index - spec.window)
    return start, index - start + 1


def build_windows(spec: MiningFactorSpec, final: pl.DataFrame, indices: list[int]) -> list[pl.DataFrame]:
    """Bounded batches of independent windows, with kernels reset per decision."""
    size = max(1, WINDOW_BATCH_CELLS // max(1, final.width - 1) // (spec.window + 1))
    rows = []
    for offset in range(0, len(indices), size):
        windows = [final.slice(*_window(spec, index)) for index in indices[offset:offset + size]]
        groups = pl.Series([group for group, frame in enumerate(windows) for _ in range(frame.height)])
        values = pl.concat(windows).select("date", spec._expression(final.columns[1:]).over(groups))
        rows.append(values.filter((groups != groups.shift(-1)).fill_null(True)))
    return rows


def build_revision_factor(
    spec: MiningFactorSpec, revisions: pl.DataFrame, decisions: pl.DataFrame,
) -> pl.DataFrame:
    """Evaluate a native factor using the versions visible at each decision.

    ``revisions`` has date (bar completion), symbol, price, recorded_at and
    known_at. ``decisions`` supplies every session's date and decision_time,
    including sessions with missing prices. All clocks must be timezone-aware.
    The latest *received eligible* revision wins for each symbol/session; ties
    are rejected. A later decision may see revised warm-up prices, while an
    earlier factor never changes. The input symbol set is a fixed study universe,
    not evidence of historical membership. This function builds features only;
    labels and portfolio valuation need their own evaluation cutoff. Output
    dates remain bar-session labels: before executing, the caller must ensure
    each decision_time precedes the actual fill opportunity. A next-row lag
    alone does not prove that for after-close decisions or close-stamped bars.
    """
    return RevisionPanel(revisions, decisions).factor(spec)


def _prepare(revisions, decisions):
    revisions = _clocks(revisions, ("date", "recorded_at", "known_at"))
    decisions = _clocks(decisions, ("date", "decision_time")).sort("date")
    if decisions.is_empty() or decisions["date"].n_unique() != decisions.height:
        raise ValueError("Decisions require a nonempty unique session grid")
    if decisions.filter(pl.col("decision_time") < pl.col("date")).height:
        raise ValueError("A completed-bar decision cannot precede its session close")
    if not decisions["decision_time"].is_sorted():
        raise ValueError("Decision times must be chronological")
    if revisions.schema.get("symbol") != pl.String or "price" not in revisions.columns:
        raise ValueError("Revisions require string symbols and numeric prices")
    if revisions["symbol"].null_count() or revisions.filter(pl.col("symbol").str.len_chars() == 0).height:
        raise ValueError("Revision symbols must be present")
    reasons = ["invalid_reason"] if "invalid_reason" in revisions.columns else []
    if reasons:
        invalid = pl.col("invalid_reason").eq("unverified_price_basis").fill_null(False)
        if revisions.filter((pl.col("price").is_null() != invalid)
                            | (pl.col("invalid_reason").is_not_null() & ~invalid)).height:
            raise ValueError("Invalid revisions require a null price and explicit supported reason")
    elif revisions["price"].null_count():
        raise ValueError("Null revision prices require an explicit invalidation reason")
    if not revisions.schema["price"].is_numeric() or revisions.filter(
        (~pl.col("price").is_finite() | (pl.col("price") <= 0)).fill_null(False)
    ).height:
        raise ValueError("Revision prices must be finite and positive")
    if revisions.select("date", "symbol", "recorded_at").is_duplicated().any():
        raise ValueError("Ambiguous received revision identity")
    if revisions.is_empty():
        raise ValueError("At least one study symbol is required")
    return revisions.select("date", "symbol", pl.col("price").cast(pl.Float64), "recorded_at", "known_at", *reasons), decisions.select("date", "decision_time")


def _membership(frame: pl.DataFrame) -> pl.DataFrame:
    if set(frame.columns) != {"symbol", "start", "end"}:
        raise ValueError("Membership requires exactly symbol, start and end")
    frame = _clocks(frame, ("start", "end")).sort("symbol", "start")
    if frame.schema["symbol"] != pl.String or frame["symbol"].null_count():
        raise ValueError("Membership symbols must be present strings")
    if frame.filter(pl.col("start") > pl.col("end")).height:
        raise ValueError("Membership intervals must not be reversed")
    return frame.select("symbol", "start", "end")


def _clocks(frame: pl.DataFrame, columns: tuple[str, ...]) -> pl.DataFrame:
    for column in columns:
        dtype = frame.schema.get(column)
        if not isinstance(dtype, pl.Datetime) or dtype.time_zone is None or frame[column].null_count():
            raise ValueError(f"{column} requires present timezone-aware timestamps")
        if dtype.time_unit == "ns" and frame.filter(pl.col(column).dt.epoch("ns") % 1000 != 0).height:
            raise ValueError(f"{column} submicrosecond timestamps cannot be represented losslessly")
    return frame.with_columns(pl.col(c).dt.convert_time_zone("UTC").cast(pl.Datetime("us", "UTC")) for c in columns)
