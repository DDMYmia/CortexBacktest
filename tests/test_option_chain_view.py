import io
import json

import polars as pl
import pytest

from cortexbacktest.cli import main
from cortexbacktest.options.chain_view import STRIKES, chain_view
from tests.test_option_strategies import synthetic_chain


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    root = tmp_path_factory.mktemp("chain-view")
    frame = synthetic_chain("SPY", 100.0)
    for (year,), part in frame.group_by(pl.col("quote_date").dt.year()):
        folder = root / "underlying=SPY" / f"year={year}"
        folder.mkdir(parents=True, exist_ok=True)
        part.drop("underlying").write_parquet(folder / "part-test.parquet")
    return root


def test_a_session_shows_its_chain_term_structure_and_summary(root):
    view = chain_view({"underlying": "spy", "date": "2024-04-14", "root": str(root)})  # a Sunday
    assert view["session"] == "2024-04-12" and view["coverage"] == ["2024-01-02", "2024-06-27"]
    assert view["dte"] >= 20 and [e["dte"] for e in view["expiries"]] == sorted(e["dte"] for e in view["expiries"])
    strikes = [row["strike"] for row in view["rows"]]
    assert strikes == sorted(strikes) and len(strikes) <= STRIKES
    summary = view["summary"]
    atm = next(row for row in view["rows"] if row["strike"] == summary["atm_strike"])
    assert summary["expected_move"] == pytest.approx(atm["call_mid"] + atm["put_mid"])
    assert atm["call_prob_itm"] + atm["put_prob_itm"] == pytest.approx(1)  # N(d2) + N(-d2)
    assert summary["put_call_oi"] == pytest.approx(1) and 0 <= summary["iv_rank"] <= 1
    assert view["iv_history"][-1] == {"date": "2024-04-12", "atm_iv": summary["iv30"]}


def test_a_chosen_expiry_and_an_unknown_underlying(root):
    expiry = chain_view({"underlying": "SPY", "date": "2024-04-12", "root": str(root)})["expiries"][0]["expiry"]
    assert chain_view({"underlying": "SPY", "date": "2024-04-12", "expiry": expiry, "root": str(root)})["expiry"] == expiry
    assert chain_view({"underlying": "XYZ", "root": str(root)}) == {"underlying": "XYZ", "coverage": None}


def test_cli_reads_the_request_on_stdin(root, monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps({"underlying": "SPY", "root": str(root)})))
    assert main(["option-chain"]) == 0
    assert json.loads(capsys.readouterr().out)["session"] == "2024-06-27"


def test_a_daily_snapshot_file_is_imported_once_and_viewable(tmp_path, capsys):
    header = ("quote_date,underlying,underlying_price,expiry,strike,right,bid,ask,implied_volatility,"
              "delta,gamma,theta,vega,volume,open_interest")
    rows = [f"2026-10-23,SPY,500,2026-11-20,{k},{r},{b},{b + 0.2},0.2,{d},0.01,-0.1,0.5,10,"
            for k in (495, 500, 505) for r, d, b in (("CALL", 0.5, 8.0), ("PUT", -0.5, 7.0))]
    rows.append("2026-10-23,SPY,500,2026-11-20,510,CALL,,,,,,,,,")  # no quote: dropped, not stored
    path = tmp_path / "SPY-2026-10-23.csv"
    path.write_text("\n".join([header, *rows]) + "\n")
    store = tmp_path / "store"
    assert main(["option-chain-import", str(path), "--root", str(store)]) == 0
    assert json.loads(capsys.readouterr().out)[path.name]["stored"] == 6
    assert main(["option-chain-import", str(path), "--root", str(store)]) == 0
    assert json.loads(capsys.readouterr().out)[path.name]["stored"] == 0  # a re-run adds nothing
    view = chain_view({"underlying": "SPY", "root": str(store)})
    assert view["session"] == "2026-10-23" and view["summary"]["atm_strike"] == 500
    assert view["rows"][1]["put_open_interest"] is None  # the delayed feed may carry no open interest
