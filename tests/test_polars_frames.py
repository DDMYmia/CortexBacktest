from __future__ import annotations

import pandas as pd
import polars as pl
import pytest

from cortexbacktest.frames import (
    factor_matrix,
    forward_returns_matrix,
    price_matrix,
    returns_matrix,
    stack_factor_and_forward_returns,
)


def _bars() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-01", "2024-01-01", "2024-01-02"],
            "symbol": ["b", "a", "b", "a"],
            "adj_close": [18.0, 10.0, 20.0, 11.0],
        }
    )


def _factors() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"],
            "symbol": ["A", "B", "A", "B"],
            "factor": [1.0, -1.0, 1.0, -1.0],
        }
    )


def test_price_and_factor_matrix_pl_are_sorted_wide_tables() -> None:
    prices = price_matrix(_bars())
    factors = factor_matrix(_factors())

    assert isinstance(prices, pl.DataFrame)
    assert prices.columns == ["date", "A", "B"]
    assert prices.select(["A", "B"]).to_dict(as_series=False) == {"A": [10.0, 11.0], "B": [20.0, 18.0]}
    assert factors.columns == ["date", "A", "B"]
    assert factors.select(["A", "B"]).to_dict(as_series=False) == {"A": [1.0, 1.0], "B": [-1.0, -1.0]}


def test_returns_matrix_pl_and_forward_returns_matrix_pl() -> None:
    prices = price_matrix(_bars())

    returns = returns_matrix(prices)
    forward = forward_returns_matrix(prices)

    assert returns["A"].to_list()[0] is None
    assert round(returns["A"].to_list()[1], 4) == 0.1
    assert round(forward["A"].to_list()[0], 4) == 0.1
    assert forward["A"].to_list()[1] is None


def test_stack_factor_and_forward_returns_pl_matches_aligned_symbols() -> None:
    factors = factor_matrix(_factors())
    forward = forward_returns_matrix(price_matrix(_bars()))

    research_frame = stack_factor_and_forward_returns(factors, forward)

    assert research_frame.columns == ["date", "symbol", "factor", "forward_return"]
    assert research_frame["symbol"].to_list() == ["A", "B"]
    assert research_frame["factor"].to_list() == [1.0, -1.0]
    assert [round(value, 4) for value in research_frame["forward_return"].to_list()] == [0.1, -0.1]


def test_polars_frames_accept_pandas_long_form_inputs() -> None:
    bars = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-01"],
            "symbol": ["A", "B"],
            "adj_close": [10.0, 20.0],
        }
    )

    prices = price_matrix(bars)

    assert isinstance(prices, pl.DataFrame)
    assert prices.columns == ["date", "A", "B"]


def test_forward_returns_matrix_pl_rejects_invalid_periods() -> None:
    with pytest.raises(ValueError, match="periods must be >= 1"):
        forward_returns_matrix(price_matrix(_bars()), periods=0)
