"""Reporting views attached to core result data."""

from dataclasses import dataclass

from ..event.result import BacktestResult as ResultData
from .result_views import ResultReportViews


@dataclass(frozen=True, slots=True, init=False)
class BacktestResult(ResultData, ResultReportViews):
    """Public result: core data with frozen field bindings and reporting views."""
