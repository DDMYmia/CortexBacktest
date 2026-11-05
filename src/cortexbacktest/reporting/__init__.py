"""Reporting, records, and analyzer subsystem."""


from .factor_risk import (
    StyleExposureReport as StyleExposureReport,
    standardise_style_exposures as standardise_style_exposures,
    portfolio_style_exposure as portfolio_style_exposure,
    factor_risk_decomposition as factor_risk_decomposition,
)

from .group_attribution import (
    GroupAttribution as GroupAttribution,
    group_attribution as group_attribution,
)

from .cashflow_returns import (
    CashflowPerformance as CashflowPerformance,
    cashflow_performance as cashflow_performance,
)

from .risk_calibration import (
    VarBacktestResult as VarBacktestResult,
    var_backtest as var_backtest,
    kupiec_pof as kupiec_pof,
    christoffersen_independence as christoffersen_independence,
)

from .analyzers import (
    trade_summary as trade_summary,
)
from .exposure import (
    exposure_frame as exposure_frame,
    exposure_summary as exposure_summary,
    group_exposure_frame as group_exposure_frame,
    positions_frame as positions_frame,
    turnover_frame as turnover_frame,
    turnover_summary as turnover_summary,
)
from .performance import (
    BenchmarkReport as BenchmarkReport,
    PerformanceReport as PerformanceReport,
    active_returns as active_returns,
    align_returns as align_returns,
    alpha as alpha,
    benchmark_report as benchmark_report,
    beta as beta,
    drawdown_series as drawdown_series,
    drawdown_table as drawdown_table,
    information_ratio as information_ratio,
    performance_report as performance_report,
    period_returns as period_returns,
    returns_from_equity as returns_from_equity,
    rolling_beta as rolling_beta,
    rolling_sharpe as rolling_sharpe,
)
from .records import (
    fills_frame as fills_frame,
    open_orders_frame as open_orders_frame,
    order_records_frame as order_records_frame,
    write_fills_csv as write_fills_csv,
    write_fills_ndjson as write_fills_ndjson,
    write_open_orders_csv as write_open_orders_csv,
    write_open_orders_ndjson as write_open_orders_ndjson,
    write_order_records_csv as write_order_records_csv,
    write_order_records_ndjson as write_order_records_ndjson,
    write_positions_csv as write_positions_csv,
    write_positions_ndjson as write_positions_ndjson,
)
from .charts import (
    chart_styles as chart_styles,
    drawdown_chart as drawdown_chart,
    equity_chart as equity_chart,
    exposure_chart as exposure_chart,
    return_histogram as return_histogram,
)
from .reports import (
    execution_sweep_report as execution_sweep_report,
    factor_sweep_report as factor_sweep_report,
    equity_curve_charts as equity_curve_charts,
    report_html as report_html,
    research_report as research_report,
    write_execution_sweep_report_html as write_execution_sweep_report_html,
    write_execution_sweep_report_json as write_execution_sweep_report_json,
    write_factor_sweep_report_html as write_factor_sweep_report_html,
    write_factor_sweep_report_json as write_factor_sweep_report_json,
    write_report_html as write_report_html,
    write_report_json as write_report_json,
    write_research_report_html as write_research_report_html,
    write_research_report_json as write_research_report_json,
)
from .summary import (
    event_report as event_report,
    event_report_html as event_report_html,
    frame_records as frame_records,
    write_event_report_html as write_event_report_html,
    write_event_report_json as write_event_report_json,
)
from .tear_sheet import (
    event_tear_sheet as event_tear_sheet,
    event_tear_sheet_html as event_tear_sheet_html,
    write_event_tear_sheet_html as write_event_tear_sheet_html,
)
from .trades import (
    pnl_reconciliation_frame as pnl_reconciliation_frame,
    pnl_reconciliation_summary as pnl_reconciliation_summary,
    trade_stats as trade_stats,
    trades_frame as trades_frame,
)

# Keep the public export list statically visible to type checkers.
__all__ = [
    "StyleExposureReport", "standardise_style_exposures", "portfolio_style_exposure", "factor_risk_decomposition",
    "GroupAttribution", "group_attribution",
    "CashflowPerformance", "cashflow_performance", "VarBacktestResult", "var_backtest", "kupiec_pof", "christoffersen_independence",
    "trade_summary", "exposure_frame", "exposure_summary", "group_exposure_frame", "positions_frame",
    "turnover_frame", "turnover_summary", "BenchmarkReport", "PerformanceReport", "active_returns",
    "align_returns", "alpha", "benchmark_report", "beta", "drawdown_series", "drawdown_table",
    "information_ratio", "performance_report", "period_returns", "returns_from_equity", "rolling_beta",
    "rolling_sharpe", "fills_frame", "open_orders_frame", "order_records_frame", "write_fills_csv",
    "write_fills_ndjson", "write_open_orders_csv", "write_open_orders_ndjson", "write_order_records_csv",
    "write_order_records_ndjson", "write_positions_csv", "write_positions_ndjson", "chart_styles",
    "drawdown_chart", "equity_chart", "exposure_chart", "return_histogram", "execution_sweep_report",
    "factor_sweep_report", "equity_curve_charts", "report_html", "research_report",
    "write_execution_sweep_report_html", "write_execution_sweep_report_json",
    "write_factor_sweep_report_html", "write_factor_sweep_report_json", "write_report_html",
    "write_report_json", "write_research_report_html", "write_research_report_json", "event_report",
    "event_report_html", "frame_records", "write_event_report_html", "write_event_report_json",
    "event_tear_sheet", "event_tear_sheet_html", "write_event_tear_sheet_html", "pnl_reconciliation_frame",
    "pnl_reconciliation_summary", "trade_stats", "trades_frame",
]
