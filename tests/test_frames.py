from datetime import datetime

import polars as pl

from cortexbacktest.frames import (
    forward_returns_matrix,
    price_matrix,
    stack_factor_and_forward_returns,
)
from cortexbacktest.research import top_bottom_weights


def test_factor_research_frame_pipeline_builds_long_form_and_weights():
    bars = pl.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"],
            "symbol": ["A", "B", "A", "B"],
            "adj_close": [10.0, 20.0, 11.0, 18.0],
        }
    )
    factors = pl.DataFrame({"date": ["2024-01-01", "2024-01-02"], "A": [1.0, 1.0], "B": [-1.0, -1.0]})

    prices = price_matrix(bars)
    forward = forward_returns_matrix(prices)
    research_frame = stack_factor_and_forward_returns(factors, forward)
    weights = top_bottom_weights(factors, top_quantile=0.5, bottom_quantile=0.5)

    assert research_frame.height == 2
    first = weights.filter(pl.col("date") == datetime(2024, 1, 1)).row(0, named=True)
    assert first["A"] == 1.0
    assert first["B"] == -1.0
