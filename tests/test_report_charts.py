"""Charts are inline SVG with no plotting dependency, and must stay deterministic."""

from datetime import datetime, timedelta

import pytest

from cortexbacktest.reporting.charts import (
    chart_styles, drawdown_chart, equity_chart, exposure_chart, return_histogram,
)
from cortexbacktest.reporting.reports import equity_curve_charts, report_html


def curve(n: int = 120) -> tuple[list[datetime], list[float]]:
    start = datetime(2024, 1, 2)
    stamps = [start + timedelta(days=i) for i in range(n)]
    equity = [1_000_000.0]
    for i in range(1, n):
        equity.append(equity[-1] * (1.01 if i % 7 else 0.97))
    return stamps, equity


@pytest.mark.parametrize("builder", [equity_chart, drawdown_chart])
def test_series_charts_are_wellformed_svg(builder):
    stamps, equity = curve()

    svg = builder(stamps, equity)

    assert svg.startswith('<svg class="cb-chart"') and svg.endswith("</svg>")
    assert 'role="img"' in svg and "aria-label=" in svg
    assert "<script" not in svg


def test_charts_are_byte_deterministic():
    stamps, equity = curve()

    assert equity_chart(stamps, equity) == equity_chart(stamps, equity)
    assert return_histogram([0.01, -0.02, 0.03] * 20) == return_histogram([0.01, -0.02, 0.03] * 20)


def test_drawdown_is_never_positive_and_the_area_closes_on_zero():
    stamps, equity = curve()

    svg = drawdown_chart(stamps, equity)

    # The frame labels the axis from the worst drawdown up to zero.
    assert ">0.0%</text>" in svg
    assert 'class="cb-loss"' in svg and 'class="cb-loss-line"' in svg


def test_degenerate_inputs_say_so_rather_than_drawing_an_empty_frame():
    assert "not enough data" in equity_chart([], [])
    assert "not enough data" in equity_chart([datetime(2024, 1, 1)], [1.0])
    assert "not enough data" in return_histogram([1.0, 1.0, 1.0])
    # A flat series still renders: the bounds are padded rather than collapsed.
    stamps = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(3)]
    assert "not enough data" not in equity_chart(stamps, [5.0, 5.0, 5.0])


def test_exposure_draws_shorts_only_when_there_are_any():
    stamps, equity = curve()
    longs = [value * 0.8 for value in equity]

    assert 'class="cb-short"' not in exposure_chart(stamps, longs, [0.0] * len(longs))
    assert 'class="cb-short"' in exposure_chart(stamps, longs, [-value * 0.2 for value in equity])


def test_equity_curve_charts_accepts_a_plain_mapping_and_skips_missing_columns():
    stamps, equity = curve(30)

    full = equity_curve_charts({"timestamp": stamps, "equity": equity,
                                "long_exposure": equity, "short_exposure": [0.0] * 30})
    minimal = equity_curve_charts({"timestamp": stamps, "equity": equity})

    assert len(full) == 4  # equity, drawdown, exposure, histogram
    assert len(minimal) == 3  # no exposure column, so no exposure chart
    assert equity_curve_charts({"timestamp": stamps}) == []


def test_report_html_embeds_the_charts_and_their_styles_once():
    stamps, equity = curve(40)
    charts = equity_curve_charts({"timestamp": stamps, "equity": equity})

    html = report_html("run", {"summary": {"final_equity": 1.0}}, charts=charts)

    assert html.count("<svg") == len(charts)
    # The style block is embedded once, however many charts it serves.
    assert html.count(".cb-chart .cb-loss-line") == 1
    assert '<section class="cb-charts">' in html
    # Still a standalone file with no runtime.
    assert "<script" not in html and "http" not in html.split("<body")[0].replace("http://www.w3.org", "")


def test_reports_are_light_only_whatever_the_viewer_scheme():
    styles = chart_styles()

    assert "prefers-color-scheme" not in styles and "dark" not in styles
    # Literal colours: no custom property a host page could redefine.
    assert "var(" not in styles and "#2f6feb" in styles and "#cb3636" in styles
