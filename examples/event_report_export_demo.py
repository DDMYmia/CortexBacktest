"""Export an event backtest report as JSON and HTML."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from cortexbacktest import (
    BacktestEngine,
    Bar,
    Order,
    Side,
    Strategy,
    write_event_report_html,
    write_event_report_json,
)


class RoundTripStrategy(Strategy):
    def on_bar(self, bar: Bar):
        if bar.timestamp.day == 1:
            return [Order(bar.symbol, bar.timestamp, Side.BUY, 10)]
        if bar.timestamp.day == 2:
            return [Order(bar.symbol, bar.timestamp, Side.SELL, 10)]
        return []


def main() -> None:
    bars = [
        Bar("AAPL", datetime(2024, 1, 1, tzinfo=UTC), 100, 100, 100, 100),
        Bar("AAPL", datetime(2024, 1, 2, tzinfo=UTC), 110, 110, 110, 110),
    ]
    result = BacktestEngine().run(bars, RoundTripStrategy())
    json_path = write_event_report_json(result, Path("examples/output/event_report.json"))
    html_path = write_event_report_html(result, Path("examples/output/event_report.html"))
    report = result.report(include_records=False)

    print("CortexBacktest - event report export demo")
    print(f"json_report: {json_path}")
    print(f"html_report: {html_path}")
    print(f"fills: {report['summary']['fills']}")
    print(f"round_trips: {report['counts']['round_trips']}")
    print(f"total_pnl: {report['trades']['total_pnl']:.2f}")


if __name__ == "__main__":
    main()
