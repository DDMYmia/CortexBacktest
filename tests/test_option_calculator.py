import io
import json

from cortexbacktest.cli import main
from cortexbacktest.options.calculator import GRID_SPOTS, TIME_LEFT, analyse

PUT = {"right": "put", "spot": 100, "strike": 100, "time": 1, "rate": 0.06, "volatility": 0.2, "american": True}


def test_every_model_prices_the_standard_american_put_and_the_grids_cover_spot_and_time():
    report = analyse({**PUT, "market_price": 6.0})
    prices = {m["model"]: m["price"] for m in report["models"]}
    for model in ("Binomial (CRR)", "Leisen-Reimer", "Trinomial", "Crank-Nicolson"):
        assert abs(prices[model] - 5.7963) < 2e-3, model
    assert abs(prices["Longstaff-Schwartz"] - 5.7963) < 0.06  # Bermudan, slightly low
    assert report["early_exercise_premium"] > 0.6
    assert report["greeks"]["delta"] < 0 and 0.2 < report["implied_volatility"] < 0.25
    assert len(report["curve"]) == GRID_SPOTS and len(report["profit"]) == GRID_SPOTS * len(TIME_LEFT)
    at_expiry = [p for p in report["profit"] if p["time_left"] == 0]
    assert all(abs(p["profit"] + report["premium"] - max(100 - p["spot"], 0)) < 1e-9 for p in at_expiry)


def test_an_impossible_market_price_is_reported_not_raised_and_cli_reads_stdin(monkeypatch, capsys):
    assert isinstance(analyse({**PUT, "american": False, "market_price": 500})["implied_volatility"], str)
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps({**PUT, "right": "CALL", "american": False})))
    assert main(["option-price"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["reference_model"] == "Black-Scholes" and report["early_exercise_premium"] == 0
