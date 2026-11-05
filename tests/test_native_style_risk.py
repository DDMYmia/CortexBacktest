from copy import deepcopy
import math
import pytest

from test_native_group_attribution import fixture
from cortexbacktest.reporting.native_style_risk import native_style_risk
from cortexbacktest.reporting.native_attribution import native_snapshot_weights


def model():
    return {
        "model_id": "declared-v1",
        "currency": "USDT",
        "horizon": "one_day",
        "fit_end": "2024-01-01T00:00:00Z",
        "model_known_at": "2024-01-02T00:00:00Z",
        "factors": ["value", "momentum"],
        "covariance": [[0.04, 0.01], [0.01, 0.09]],
        "assets": [
            {"instrument_id": 1, "loadings": [1.0, 2.0], "specific_variance": 0.01},
            {"instrument_id": 2, "loadings": [3.0, 4.0], "specific_variance": 0.04},
        ],
    }


def report(config=None):
    runs, latest, _ = fixture()
    return native_style_risk(runs, latest, config or model())


def test_actual_closing_weights_style_and_independent_risk_oracle():
    runs, latest, _ = fixture()
    weights = native_snapshot_weights(runs["qlib"], latest)
    assert weights.filter(weights["symbol"] == "__CASH__")["weight"].to_list() == pytest.approx(
        [1.0, 48.48 / 103.48, 1.0]
    )
    result = report()
    assert result["snapshotCount"] == 3
    value = [r for r in result["style"]["history"] if r["factor"] == "value"]
    assert [r["portfolio_exposure"] for r in value] == pytest.approx([0.0, 55 / 103.48, 0.0])
    assert [r["benchmark_exposure"] for r in value] == pytest.approx([0.0, 1.5, 1.5])
    # Benchmark final factor exposure=[1.5,2], variance=.51 plus specific=.01.
    risk = result["risk"]["benchmark"]
    assert risk["factorVariance"] == pytest.approx(0.51)
    assert risk["specificVariance"] == pytest.approx(0.01)
    assert risk["volatility"] == pytest.approx(math.sqrt(0.52))
    assert sum(a["volatilityContribution"] for a in risk["assets"]) == pytest.approx(
        math.sqrt(0.52)
    )
    assert result["risk"]["active"]["totalVariance"] == pytest.approx(0.52)
    assert result["risk"]["portfolio"]["riskShareStatus"] == "undefined_zero_risk"
    assert all(f["riskShare"] is None for f in result["risk"]["portfolio"]["factors"])
    assert not result["qualificationIssued"] and not result["executionAuthorized"]


def test_factor_order_permutation_preserves_named_risk_and_style_values():
    original = report()
    changed = model()
    changed["factors"].reverse()
    changed["covariance"] = [row[::-1] for row in changed["covariance"][::-1]]
    for asset in changed["assets"]:
        asset["loadings"].reverse()
    alternate = report(changed)
    for key in ["portfolio", "benchmark", "active"]:
        assert alternate["risk"][key]["totalVariance"] == pytest.approx(
            original["risk"][key]["totalVariance"]
        )
    assert alternate["style"]["history"] == original["style"]["history"]


@pytest.mark.parametrize(
    "patch",
    [
        {"model_known_at": "2025-01-02T00:00:00Z"},
        {"model_known_at": True},
        {"fit_end": "2024-01-03T00:00:00Z"},
        {"covariance": [[1.0, 2.0], [2.0, 1.0]]},
        {"covariance": [[1.0, 0.1], [0.2, 1.0]]},
        {"covariance": [[1.0]]},
        {"factors": ["value", "value"]},
        {"assets": [{"instrument_id": 2, "loadings": [3.0, 4.0], "specific_variance": 0.04}]},
        {
            "assets": [
                {"instrument_id": 1, "loadings": [1.0, 2.0], "specific_variance": -0.01},
                {"instrument_id": 2, "loadings": [3.0, 4.0], "specific_variance": 0.04},
            ]
        },
    ],
)
def test_bad_or_future_incomplete_models_never_default_to_zero(patch):
    with pytest.raises(ValueError):
        report({**model(), **deepcopy(patch)})


def test_zero_covariance_and_specific_variance_keep_null_risk_shares():
    config = model()
    config["covariance"] = [[0.0, 0.0], [0.0, 0.0]]
    for asset in config["assets"]:
        asset["specific_variance"] = 0.0
    result = report(config)
    assert all(
        v["volatility"] == 0.0 and v["riskShareStatus"] == "undefined_zero_risk"
        for v in result["risk"].values()
    )
