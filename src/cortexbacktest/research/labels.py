"""Forward labels use the signal universe and a separate holding-period price set."""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl


@dataclass(frozen=True, slots=True)
class ForwardLabelResult:
    panel: pl.DataFrame
    coverage: dict[str, object]


def attach_forward_labels(
    decisions: pl.DataFrame,
    prices: pl.DataFrame,
    sessions: pl.Series,
    *,
    price_basis: str,
    price_data_id: str,
    entry_lag: int = 1,
    holding_sessions: int = 1,
) -> ForwardLabelResult:
    """Join entry/exit opens by canonical instrument and explicit session date.

    Prices must cover holdings even after they leave the next signal universe.
    The caller supplies verified total-return-adjusted opens and their dataset
    identity; this declaration does not itself verify adjustment provenance.
    Missing/invalid quotes stay null and are counted, never dropped or filled.
    Terminal cash-outs require separately evidenced outcomes, not zero quotes.
    """
    if price_basis != "total_return_adjusted" or not price_data_id.strip():
        raise ValueError("verified total_return_adjusted prices and price_data_id are required")
    if type(entry_lag) is not int or type(holding_sessions) is not int or min(entry_lag, holding_sessions) < 1:
        raise ValueError("entry lag and holding sessions must be positive integers")
    if not {"date", "instrument_id"} <= set(decisions.columns):
        raise ValueError("decisions require date and instrument_id")
    if not {"date", "instrument_id", "open"} <= set(prices.columns):
        raise ValueError("prices require date, instrument_id and open")
    outputs = {"entry_date", "exit_date", "entry_open", "exit_open", "forward_return"}
    if outputs & set(decisions.columns):
        raise ValueError("remove existing label columns explicitly before attaching new labels")
    if sessions.is_empty() or sessions.null_count() or sessions.is_duplicated().any():
        raise ValueError("sessions must be nonempty, nonnull and unique")
    for frame in (decisions, prices):
        if frame.select(pl.any_horizontal(pl.col("date", "instrument_id").is_null()).any()).item():
            raise ValueError("date/instrument identity must not be null")
        if frame.select(pl.struct("date", "instrument_id").is_duplicated().any()).item():
            raise ValueError("duplicate date/instrument identity")
    calendar = pl.DataFrame({"date": sessions.sort()}).with_columns(
        pl.col("date").shift(-entry_lag).alias("entry_date"),
        pl.col("date").shift(-entry_lag - holding_sessions).alias("exit_date"),
    )
    if decisions.join(calendar.select("date"), on="date", how="anti").height:
        raise ValueError("decision date is absent from the session calendar")
    result = decisions.join(calendar, on="date", how="left", maintain_order="left")
    for side in ("entry", "exit"):
        quotes = prices.select(
            "instrument_id", pl.col("date").alias(f"{side}_date"),
            pl.when(pl.col("open").is_finite() & (pl.col("open") > 0))
            .then(pl.col("open")).alias(f"{side}_open"),
        )
        result = result.join(quotes, on=["instrument_id", f"{side}_date"],
                             how="left", maintain_order="left")
    result = result.with_columns(
        (pl.col("exit_open") / pl.col("entry_open") - 1).alias("forward_return"))
    result = result.with_columns(pl.when(pl.col("forward_return").is_finite())
                                 .then(pl.col("forward_return")).alias("forward_return"))
    censored = result["exit_date"].null_count()
    return ForwardLabelResult(result, {
        "price_basis": price_basis, "price_data_id": price_data_id,
        "return_kind": "simple_total_return", "entry_lag": entry_lag,
        "holding_sessions": holding_sessions, "label_horizon_sessions": entry_lag + holding_sessions,
        "decision_rows": decisions.height, "right_censored_rows": censored,
        "unresolved_rows": result["forward_return"].null_count() - censored,
        "valid_rows": result["forward_return"].count(),
    })
