import json

import pandas as pd
import pytest

from cortexbacktest import load_backtest_dataset, load_target_weights, save_backtest_dataset
from cortexbacktest.cli import main


def _bars() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": [
                "2024-01-01",
                "2024-01-01",
                "2024-01-02",
                "2024-01-02",
                "2024-01-03",
                "2024-01-03",
                "2024-01-04",
                "2024-01-04",
            ],
            "symbol": ["A", "B", "A", "B", "A", "B", "A", "B"],
            "adj_close": [10, 20, 11, 19, 12, 18, 13, 17],
        }
    )


def _factors() -> pd.DataFrame:
    # The last bar date supplies returns after the three factor dates.
    return _bars()[["date", "symbol"]].head(6).assign(factor=[1.0, -1.0] * 3)


def _dataset_dir(tmp_path):
    dataset = load_backtest_dataset(bars=_bars(), factors=_factors())
    return save_backtest_dataset(dataset, tmp_path / "dataset")


def test_dataset_inspect_cli_outputs_dataset_summary(tmp_path, capsys):
    dataset_dir = _dataset_dir(tmp_path)

    status = main(["dataset-inspect", "--dataset", str(dataset_dir)])

    assert status == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["bars"]["rows"] == 8
    assert payload["bars"]["symbols"] == 2
    assert payload["matrices"]["prices"] == [4, 2]
    assert payload["quality"]["ok"]


def test_dataset_build_cli_creates_standard_dataset_directory(tmp_path, capsys):
    bars_path = tmp_path / "bars.csv"
    factors_path = tmp_path / "factors.csv"
    output_dir = tmp_path / "built_dataset"
    _bars().to_csv(bars_path, index=False)
    _factors().to_csv(factors_path, index=False)

    status = main(
        [
            "dataset-build",
            "--bars",
            str(bars_path),
            "--factors",
            str(factors_path),
            "--output",
            str(output_dir),
        ]
    )

    assert status == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["output"] == str(output_dir)
    assert payload["quality"]["ok"]
    assert payload["matrices"]["prices"] == [4, 2]
    assert (output_dir / "manifest.json").exists()
    assert (output_dir / "bars.csv").exists()
    assert (output_dir / "factors.csv").exists()

    inspect_status = main(["dataset-inspect", "--dataset", str(output_dir)])
    assert inspect_status == 0
    inspect_payload = json.loads(capsys.readouterr().out)
    assert inspect_payload["bars"]["rows"] == 8


def test_dataset_validate_cli_returns_zero_for_valid_files(tmp_path, capsys):
    bars_path = tmp_path / "bars.csv"
    factors_path = tmp_path / "factors.csv"
    _bars().to_csv(bars_path, index=False)
    _factors().to_csv(factors_path, index=False)

    status = main(
        [
            "dataset-validate",
            "--bars",
            str(bars_path),
            "--factors",
            str(factors_path),
        ]
    )

    assert status == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"]
    assert payload["dataset"]["prices_shape"] == [4, 2]


def test_dataset_validate_cli_returns_nonzero_for_bad_bars(tmp_path, capsys):
    bars_path = tmp_path / "bad_bars.csv"
    bad = pd.concat([_bars(), _bars().iloc[[0]]], ignore_index=True)
    bad.loc[0, "adj_close"] = 0
    bad.to_csv(bars_path, index=False)

    status = main(["dataset-validate", "--bars", str(bars_path)])

    assert status == 1
    payload = json.loads(capsys.readouterr().out)
    assert not payload["ok"]
    assert any("duplicate date/symbol" in error for error in payload["errors"])
    assert any("adj_close contains" in error for error in payload["errors"])


def test_factor_research_cli_writes_reports(tmp_path, capsys):
    dataset_dir = _dataset_dir(tmp_path)
    json_path = tmp_path / "research.json"
    html_path = tmp_path / "research.html"

    status = main(
        [
            "factor-research",
            "--dataset",
            str(dataset_dir),
            "--output-json",
            str(json_path),
            "--output-html",
            str(html_path),
            "--horizons",
            "1",
            "--quantiles",
            "2",
        ]
    )

    assert status == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["settings"]["horizons"] == [1]
    assert payload["outputs"]["json"] == str(json_path)
    assert json.loads(json_path.read_text())["summary"]
    assert "Research Report" in html_path.read_text()


def test_factor_research_cli_accepts_preprocessing_options(tmp_path, capsys):
    dataset_dir = _dataset_dir(tmp_path)
    groups_path = tmp_path / "groups.csv"
    json_path = tmp_path / "research_preprocessed.json"
    pd.DataFrame(
        {
            "symbol": ["A", "B"],
            "group": ["technology", "defensive"],
        }
    ).to_csv(groups_path, index=False)

    status = main(
        [
            "factor-research",
            "--dataset",
            str(dataset_dir),
            "--output-json",
            str(json_path),
            "--horizons",
            "1",
            "--quantiles",
            "2",
            "--winsorize",
            "0.05,0.95",
            "--rank",
            "--zscore",
            "--demean",
            "--neutralize-groups-file",
            str(groups_path),
        ]
    )

    assert status == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["settings"]["preprocessing"]["winsorize"] == [0.05, 0.95]
    assert payload["settings"]["preprocessing"]["rank"]
    assert payload["settings"]["preprocessing"]["zscore"]
    assert payload["settings"]["preprocessing"]["demean"]
    assert payload["settings"]["preprocessing"]["neutralize_groups_file"] == str(groups_path)
    assert json.loads(json_path.read_text())["summary"]


@pytest.mark.parametrize(("command", "output", "option", "value", "message"), [
    ("factor-research", "research.json", "--horizons", "0", "horizons must be positive integers"),
    ("factor-research", "research.json", "--winsorize", "0.01", "winsorize requires 2 comma-separated values"),
    ("event-backtest", "event.json", "--max-group-weight", "technology=0.1", "--max-group-weight requires --groups-file"),
])
def test_cli_rejects_invalid_options(tmp_path, capsys, command, output, option, value, message):
    status = main([command, "--dataset", str(_dataset_dir(tmp_path)), "--output-json", str(tmp_path / output), option, value])
    assert status == 1
    assert message in capsys.readouterr().err


def test_factor_sweep_cli_writes_reports(tmp_path, capsys):
    dataset_dir = _dataset_dir(tmp_path)
    json_path = tmp_path / "factor_sweep.json"
    html_path = tmp_path / "factor_sweep.html"

    status = main(
        [
            "factor-sweep",
            "--dataset",
            str(dataset_dir),
            "--case",
            "mom_1=momentum:window=1",
            "--case",
            "mom_2=momentum:window=2",
            "--output-json",
            str(json_path),
            "--output-html",
            str(html_path),
            "--horizons",
            "1",
            "--quantiles",
            "2",
            "--zscore",
        ]
    )

    assert status == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["settings"]["cases"] == ["mom_1", "mom_2"]
    assert payload["settings"]["preprocessing"]["zscore"]
    assert "best_case" in payload
    report = json.loads(json_path.read_text())
    assert report["summary"]
    assert "Factor Sweep Report" in html_path.read_text()


def test_factor_sweep_cli_rejects_unknown_indicator(tmp_path, capsys):
    dataset_dir = _dataset_dir(tmp_path)

    status = main(
        [
            "factor-sweep",
            "--dataset",
            str(dataset_dir),
            "--case",
            "bad=unknown:window=1",
            "--output-json",
            str(tmp_path / "factor_sweep.json"),
        ]
    )

    captured = capsys.readouterr()
    assert status == 1
    assert "Unsupported factor indicator" in captured.err


def test_execution_sweep_cli_writes_reports(tmp_path, capsys):
    dataset_dir = _dataset_dir(tmp_path)
    json_path = tmp_path / "execution_sweep.json"
    html_path = tmp_path / "execution_sweep.html"

    status = main(
        [
            "execution-sweep",
            "--dataset",
            str(dataset_dir),
            "--case",
            "mom_1=momentum:window=1",
            "--case",
            "mom_2=momentum:window=2",
            "--output-json",
            str(json_path),
            "--output-html",
            str(html_path),
            "--horizons",
            "1",
            "--quantiles",
            "2",
            "--top-quantile",
            "0.5",
            "--bottom-quantile",
            "0",
            "--min-trade-value",
            "1",
            "--commission-bps",
            "1",
            "--max-workers",
            "2",
            "--parallel-backend",
            "process",
        ]
    )

    assert status == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["settings"]["cases"] == ["mom_1", "mom_2"]
    assert payload["settings"]["commission_bps"] == 1.0
    assert payload["settings"]["max_workers"] == 2
    assert payload["settings"]["parallel_backend"] == "process"
    assert "best_case" in payload
    report = json.loads(json_path.read_text())
    assert report["summary"]
    assert "Execution Sweep Report" in html_path.read_text()


def test_research_execution_cli_writes_pipeline_reports(tmp_path, capsys):
    dataset_dir = _dataset_dir(tmp_path)
    json_path = tmp_path / "research_execution.json"
    html_path = tmp_path / "research_execution.html"

    status = main(
        [
            "research-execution",
            "--dataset",
            str(dataset_dir),
            "--output-json",
            str(json_path),
            "--output-html",
            str(html_path),
            "--horizons",
            "1",
            "--quantiles",
            "2",
            "--top-quantile",
            "0.5",
            "--bottom-quantile",
            "0",
            "--min-trade-value",
            "1",
            "--commission-bps",
            "1",
        ]
    )

    assert status == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["settings"]["horizons"] == [1]
    assert payload["settings"]["bottom_quantile"] is None
    assert payload["settings"]["commission_bps"] == 1.0
    assert payload["summary"]["event_final_equity"] > 100_000
    assert payload["summary"]["fills"] >= 1
    report = json.loads(json_path.read_text())
    assert set(report) == {"event", "research", "summary", "target_weights"}
    assert "Research Execution Pipeline" in html_path.read_text()


def test_research_execution_cli_applies_risk_rules(tmp_path, capsys):
    dataset_dir = _dataset_dir(tmp_path)
    json_path = tmp_path / "research_execution_risk.json"

    status = main(
        [
            "research-execution",
            "--dataset",
            str(dataset_dir),
            "--output-json",
            str(json_path),
            "--horizons",
            "1",
            "--quantiles",
            "2",
            "--top-quantile",
            "0.5",
            "--bottom-quantile",
            "0",
            "--max-order-value",
            "100",
        ]
    )

    assert status == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["settings"]["risk_rules"]["max_order_value"] == 100.0
    report = json.loads(json_path.read_text())
    assert report["event"]["counts"]["fills"] == 0
    assert report["event"]["counts"]["orders"] >= 1


def test_event_backtest_cli_writes_reports(tmp_path, capsys):
    dataset_dir = _dataset_dir(tmp_path)
    json_path = tmp_path / "event.json"
    html_path = tmp_path / "event.html"
    records_dir = tmp_path / "event_records"

    status = main(
        [
            "event-backtest",
            "--trade-on-close",
            "--fill-price",
            "close",
            "--dataset",
            str(dataset_dir),
            "--output-json",
            str(json_path),
            "--output-html",
            str(html_path),
            "--records-dir",
            str(records_dir),
            "--records-format",
            "ndjson",
            "--top-quantile",
            "0.5",
            "--bottom-quantile",
            "0",
            "--min-trade-value",
            "1",
        ]
    )

    assert status == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["settings"]["bottom_quantile"] is None
    assert payload["outputs"]["json"] == str(json_path)
    assert payload["outputs"]["records_dir"] == str(records_dir)
    assert payload["settings"]["records_format"] == "ndjson"
    report = json.loads(json_path.read_text())
    assert report["summary"]["final_equity"] > 100_000
    assert report["counts"]["fills"] >= 1
    assert "CortexBacktest Report" in html_path.read_text()
    assert (records_dir / "orders.ndjson").exists()
    assert (records_dir / "fills.ndjson").exists()


def test_event_backtest_cli_supports_summary_result_retention(tmp_path, capsys):
    dataset_dir = _dataset_dir(tmp_path)
    json_path = tmp_path / "event_summary.json"

    status = main(
        [
            "event-backtest",
            "--trade-on-close",
            "--fill-price",
            "close",
            "--dataset",
            str(dataset_dir),
            "--output-json",
            str(json_path),
            "--result-retention",
            "summary",
            "--top-quantile",
            "0.5",
            "--bottom-quantile",
            "0",
            "--min-trade-value",
            "1",
        ]
    )

    assert status == 0
    payload = json.loads(capsys.readouterr().out)
    report = json.loads(json_path.read_text())
    assert payload["settings"]["result_retention"] == "summary"
    assert payload["settings"]["include_records"] is False
    assert report["records_retained"] is False
    assert report["counts"]["fills"] >= 1
    assert "records" not in report


def test_event_backtest_cli_streams_records_with_summary_retention(tmp_path, capsys):
    dataset_dir = _dataset_dir(tmp_path)
    json_path = tmp_path / "event_summary_records.json"
    records_dir = tmp_path / "summary_records"

    status = main(
        [
            "event-backtest",
            "--dataset",
            str(dataset_dir),
            "--output-json",
            str(json_path),
            "--result-retention",
            "summary",
            "--records-dir",
            str(records_dir),
            "--records-format",
            "ndjson",
            "--top-quantile",
            "0.5",
            "--bottom-quantile",
            "0",
            "--min-trade-value",
            "1",
        ]
    )

    assert status == 0
    payload = json.loads(capsys.readouterr().out)
    report = json.loads(json_path.read_text())
    assert payload["outputs"]["records_dir"] == str(records_dir)
    assert payload["settings"]["result_retention"] == "summary"
    assert report["records_retained"] is False
    assert (records_dir / "orders.ndjson").exists()
    assert (records_dir / "fills.ndjson").exists()
    assert (records_dir / "equity_curve.ndjson").exists()


def test_event_backtest_cli_includes_benchmark_report(tmp_path, capsys):
    dataset_dir = _dataset_dir(tmp_path)
    benchmark_path = tmp_path / "benchmark.csv"
    json_path = tmp_path / "event_with_benchmark.json"
    html_path = tmp_path / "event_with_benchmark.html"
    pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
            "equity": [100_000.0, 100_200.0, 100_100.0, 100_300.0],
        }
    ).to_csv(benchmark_path, index=False)

    status = main(
        [
            "event-backtest",
            "--dataset",
            str(dataset_dir),
            "--output-json",
            str(json_path),
            "--output-html",
            str(html_path),
            "--top-quantile",
            "0.5",
            "--bottom-quantile",
            "0",
            "--min-trade-value",
            "1",
            "--benchmark",
            str(benchmark_path),
            "--benchmark-rolling-window",
            "2",
        ]
    )

    assert status == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["settings"]["benchmark"] == str(benchmark_path)
    assert "benchmark" in payload
    report = json.loads(json_path.read_text())
    assert "benchmark" in report
    assert "active_total_return" in report["benchmark"]["summary"]
    assert "Benchmark" in html_path.read_text()


def test_load_target_weights_supports_long_form_csv(tmp_path):
    weights_path = tmp_path / "weights.csv"
    pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-01", "2024-01-02"],
            "symbol": ["a", "b", "a"],
            "weight": [1.0, 0.0, 1.0],
        }
    ).to_csv(weights_path, index=False)

    weights = load_target_weights(weights_path)

    assert weights.columns == ["date", "A", "B"]
    assert weights["A"].to_list() == [1.0, 1.0]
    assert weights["B"].to_list() == [0.0, None]


def test_load_target_weights_supports_wide_csv(tmp_path):
    weights_path = tmp_path / "weights.csv"
    pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02"],
            "a": [1.0, 1.0],
            "b": [0.0, 0.0],
        }
    ).to_csv(weights_path, index=False)

    weights = load_target_weights(weights_path)

    assert weights.columns == ["date", "A", "B"]
    assert weights["A"].to_list() == [1.0, 1.0]


def test_event_backtest_cli_accepts_target_weights_file(tmp_path, capsys):
    dataset_dir = _dataset_dir(tmp_path)
    weights_path = tmp_path / "weights.csv"
    json_path = tmp_path / "event_from_weights.json"
    pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "A": [1.0, 1.0, 1.0],
            "B": [0.0, 0.0, 0.0],
        }
    ).to_csv(weights_path, index=False)

    status = main(
        [
            "event-backtest",
            "--trade-on-close",
            "--fill-price",
            "close",
            "--dataset",
            str(dataset_dir),
            "--target-weights",
            str(weights_path),
            "--output-json",
            str(json_path),
            "--min-trade-value",
            "1",
        ]
    )

    assert status == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["settings"]["target_weights"] == str(weights_path)
    assert json.loads(json_path.read_text())["counts"]["fills"] >= 1


def test_event_backtest_cli_loads_user_strategy_file(tmp_path, capsys):
    dataset_dir = _dataset_dir(tmp_path)
    strategy_path = tmp_path / "user_strategy.py"
    json_path = tmp_path / "user_strategy_event.json"
    strategy_path.write_text(
        """
from cortexbacktest import Order, Side, Strategy


class BuyOnceStrategy(Strategy):
    def __init__(self, symbol="A"):
        self.symbol = symbol
        self.bought = False

    def on_bar(self, bar):
        if self.bought or bar.symbol != self.symbol:
            return []
        self.bought = True
        return [Order(symbol=bar.symbol, timestamp=bar.timestamp, side=Side.BUY, quantity=10)]
""".lstrip(),
        encoding="utf-8",
    )

    status = main(
        [
            "event-backtest",
            "--dataset",
            str(dataset_dir),
            "--strategy",
            f"{strategy_path}:BuyOnceStrategy",
            "--strategy-kwargs",
            '{"symbol": "A"}',
            "--output-json",
            str(json_path),
        ]
    )

    assert status == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["settings"]["strategy"] == f"{strategy_path}:BuyOnceStrategy"
    report = json.loads(json_path.read_text())
    assert report["counts"]["fills"] == 1
    assert report["summary"]["final_equity"] > 100_000


def test_event_backtest_cli_rejects_strategy_with_target_weights(tmp_path, capsys):
    dataset_dir = _dataset_dir(tmp_path)
    strategy_path = tmp_path / "user_strategy.py"
    weights_path = tmp_path / "weights.csv"
    strategy_path.write_text(
        "from cortexbacktest import Strategy\nclass EmptyStrategy(Strategy):\n    pass\n",
        encoding="utf-8",
    )
    pd.DataFrame({"date": ["2024-01-01"], "A": [1.0]}).to_csv(weights_path, index=False)

    status = main(
        [
            "event-backtest",
            "--dataset",
            str(dataset_dir),
            "--strategy",
            f"{strategy_path}:EmptyStrategy",
            "--target-weights",
            str(weights_path),
            "--output-json",
            str(tmp_path / "event.json"),
        ]
    )

    captured = capsys.readouterr()
    assert status == 1
    assert "--strategy and --target-weights cannot be used together" in captured.err


def test_event_backtest_cli_passes_broker_execution_settings(tmp_path, capsys):
    dataset_dir = _dataset_dir(tmp_path)
    json_path = tmp_path / "event_with_broker_settings.json"

    status = main(
        [
            "event-backtest",
            "--dataset",
            str(dataset_dir),
            "--output-json",
            str(json_path),
            "--top-quantile",
            "0.5",
            "--bottom-quantile",
            "0",
            "--fill-price",
            "open",
            "--no-allow-short",
            "--max-volume-share",
            "0.5",
        ]
    )

    assert status == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["settings"]["fill_price"] == "open"
    assert not payload["settings"]["allow_short"]
    assert payload["settings"]["max_volume_share"] == 0.5


def test_event_backtest_cli_applies_risk_rules(tmp_path, capsys):
    dataset_dir = _dataset_dir(tmp_path)
    json_path = tmp_path / "event_with_risk_rules.json"

    status = main(
        [
            "event-backtest",
            "--dataset",
            str(dataset_dir),
            "--output-json",
            str(json_path),
            "--top-quantile",
            "0.5",
            "--bottom-quantile",
            "0",
            "--max-order-value",
            "100",
        ]
    )

    assert status == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["settings"]["risk_rules"]["max_order_value"] == 100.0
    report = json.loads(json_path.read_text())
    assert report["counts"]["fills"] == 0
    assert report["counts"]["orders"] >= 1


def test_event_backtest_cli_applies_group_risk_rule(tmp_path, capsys):
    dataset_dir = _dataset_dir(tmp_path)
    groups_path = tmp_path / "groups.csv"
    json_path = tmp_path / "event_with_group_risk.json"
    pd.DataFrame(
        {
            "symbol": ["A", "B"],
            "group": ["technology", "defensive"],
        }
    ).to_csv(groups_path, index=False)

    status = main(
        [
            "event-backtest",
            "--dataset",
            str(dataset_dir),
            "--output-json",
            str(json_path),
            "--top-quantile",
            "0.5",
            "--bottom-quantile",
            "0",
            "--groups-file",
            str(groups_path),
            "--max-group-weight",
            "technology=0.1",
        ]
    )

    assert status == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["settings"]["risk_rules"]["groups_file"] == str(groups_path)
    assert payload["settings"]["risk_rules"]["max_group_weight"] == {"technology": 0.1}
    report = json.loads(json_path.read_text())
    assert report["counts"]["fills"] == 0


@pytest.mark.parametrize("manifest", [False, True])
def test_event_cli_manifest_is_explicit_opt_in(tmp_path, capsys, manifest):
    output = tmp_path / "event.json"
    arguments = [
        "event-backtest", "--dataset", str(_dataset_dir(tmp_path)),
        "--output-json", str(output), "--top-quantile", "0.5", "--bottom-quantile", "0",
    ]
    if manifest:
        arguments.append("--build-manifest")
    assert main(arguments) == 0
    capsys.readouterr()
    report = json.loads(output.read_text())
    assert (report.get("run_manifest") is not None) is manifest
