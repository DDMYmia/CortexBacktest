"""The v1 API surface is a contract, so it is pinned rather than inferred."""

import cortexbacktest
import cortexbacktest.api as api


EXPECTED_V1 = {
    "API_VERSION", "BacktestConfig", "BrokerConfig", "StrategySpec", "run_backtest",
    "Bar", "BacktestResult", "Strategy", "StrategyFactory",
    "FillPrice", "ResultRetention", "SnapshotFrequency",
    "CashInterestModel", "CommissionModel", "CorporateAction", "EventAnalyzer",
    "MatchingModel", "RiskRule", "SettlementModel", "ShortBorrowFeeModel", "SlippageModel",
}


def test_v1_surface_is_exactly_what_is_documented():
    # Changing this set is an API version decision, not a refactor. If a name
    # belongs in tier 1, add it here and to docs/api.md together.
    assert set(api.__all__) == EXPECTED_V1
    assert api.API_VERSION == "v1"


def test_every_v1_name_resolves():
    for name in api.__all__:
        assert getattr(api, name, None) is not None, name


def test_v1_does_not_leak_implementation_imports():
    # `api` had no __all__ once, so `Any`, `Callable`, `Mapping`, `dataclass`
    # and `field` were part of its apparent surface. They are not API.
    leaked = {"Any", "Callable", "Iterable", "Literal", "Mapping",
              "MappingProxyType", "annotations", "dataclass", "field"}
    assert leaked.isdisjoint(api.__all__)


def test_the_extension_layer_is_reachable_but_not_tier_one():
    # Deliberately outside the v1 contract: reaching for these means leaving the
    # configuration API, and that should be a visible choice.
    for name in ("BacktestEngine", "BrokerSimulator", "ColumnarBarFeed"):
        assert name not in api.__all__
        assert hasattr(cortexbacktest, name), f"{name} must stay publicly reachable"


def test_the_root_carries_both_tiers():
    assert EXPECTED_V1 <= set(cortexbacktest.__all__)
