"""Stable, configuration-first public API for standard event backtests.

Use this module for new integrations. Lower-level event classes remain public
for advanced matching, streaming sinks, and custom broker behavior.

`__all__` is the v1 contract, and it is enumerated rather than left to whatever
happens to be importable: without it this module's surface included `Any`,
`Callable`, `Mapping`, `dataclass` and `field`, which are implementation imports
and were never meant to be part of anything. What is listed here is what
`API_VERSION` versions.

Deliberately absent: `BacktestEngine`, `BrokerSimulator` and `ColumnarBarFeed`.
They are imported below because the configuration layer builds on them, and they
remain public from `cortexbacktest` and `cortexbacktest.event` -- but they are the
extension layer, and a caller reaching for them has left the golden path and
should know it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Callable, Iterable, Literal, Mapping

from .contracts import Bar, CorporateAction, FillPrice
from .event.analyzers import EventAnalyzer
from .event.broker import BrokerSimulator
from .event.costs import CommissionModel, SlippageModel
from .event.engine import BacktestEngine, BacktestResult, ResultRetention, SnapshotFrequency
from .event.feed import ColumnarBarFeed
from .event.financing import CashInterestModel, SettlementModel, ShortBorrowFeeModel
from .event.matching import MatchingModel
from .event.risk import RiskRule
from .event.strategy import Strategy

API_VERSION = "v1"
StrategyFactory = Callable[..., Any]

__all__ = [
    # The entry point and its configuration.
    "API_VERSION",
    "BacktestConfig",
    "BrokerConfig",
    "StrategySpec",
    "run_backtest",
    # What a caller writes and receives.
    "Bar",
    "BacktestResult",
    "Strategy",
    "StrategyFactory",
    # Values that appear in configuration.
    "FillPrice",
    "ResultRetention",
    "SnapshotFrequency",
    # Pluggable models a configuration may carry.
    "CashInterestModel",
    "CommissionModel",
    "CorporateAction",
    "EventAnalyzer",
    "MatchingModel",
    "RiskRule",
    "SettlementModel",
    "ShortBorrowFeeModel",
    "SlippageModel",
]


@dataclass(frozen=True, slots=True)
class BrokerConfig:
    """Broker settings for the stable API.

    The default is a cash-funded, long-only account. Opt into shorting or
    same-bar close execution only when those choices are intentional.
    """

    initial_cash: float = 100_000.0
    commission_bps: float = 0.0
    slippage_bps: float = 0.0
    commission_model: CommissionModel | None = None
    slippage_model: SlippageModel | None = None
    matching_model: MatchingModel | None = None
    settlement_model: SettlementModel | None = None
    cash_interest_model: CashInterestModel | None = None
    short_borrow_fee_model: ShortBorrowFeeModel | None = None
    contract_multipliers: Mapping[str, float] = field(default_factory=dict)
    # Symbols settled by variation margin rather than by exchanging principal.
    futures_symbols: tuple[str, ...] = ()
    # Liquidate down to this fraction of notional when equity falls short of it.
    # None leaves an underwater account running, which is what every backtest
    # written before this option existed assumes.
    maintenance_margin: float | None = None
    # Default fraction withheld from a received dividend; a CorporateAction may
    # override it per issuer.
    dividend_withholding_tax: float = 0.0
    corporate_actions: tuple[CorporateAction, ...] = ()
    intrabar_policy: Literal["submission", "stop_first", "limit_first"] = "submission"
    stop_limit_same_bar: bool = True
    fill_price: FillPrice = FillPrice.OPEN
    allow_short: bool = False
    max_volume_share: float = 1.0
    risk_rules: tuple[RiskRule, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "contract_multipliers", MappingProxyType(dict(self.contract_multipliers)))
        object.__setattr__(self, "risk_rules", tuple(self.risk_rules))
        object.__setattr__(self, "corporate_actions", tuple(self.corporate_actions))
        object.__setattr__(self, "futures_symbols", tuple(self.futures_symbols))

    def create_broker(self) -> BrokerSimulator:
        """Create an independent broker; configurations never share account state."""
        return BrokerSimulator(
            initial_cash=self.initial_cash,
            commission_bps=self.commission_bps,
            slippage_bps=self.slippage_bps,
            commission_model=self.commission_model,
            slippage_model=self.slippage_model,
            matching_model=self.matching_model,
            settlement_model=self.settlement_model,
            cash_interest_model=self.cash_interest_model,
            short_borrow_fee_model=self.short_borrow_fee_model,
            contract_multipliers=dict(self.contract_multipliers),
            futures_symbols=self.futures_symbols,
            maintenance_margin=self.maintenance_margin,
            dividend_withholding_tax=self.dividend_withholding_tax,
            corporate_actions=self.corporate_actions,
            intrabar_policy=self.intrabar_policy,
            stop_limit_same_bar=self.stop_limit_same_bar,
            fill_price=self.fill_price,
            allow_short=self.allow_short,
            max_volume_share=self.max_volume_share,
            risk_rules=list(self.risk_rules),
        )


@dataclass(frozen=True, slots=True)
class BacktestConfig:
    """Complete standard-run configuration with safe execution defaults.

    `trade_on_close=False` means a strategy signal produced from bar T is first
    eligible on a later bar. With the default `BrokerConfig.fill_price=OPEN`,
    ordinary market orders fill at the next bar open.
    """

    broker: BrokerConfig = field(default_factory=BrokerConfig)
    trade_on_close: bool = False
    snapshot_frequency: SnapshotFrequency = "bar"
    result_retention: ResultRetention = "full"
    build_manifest: bool = True
    analyzers: tuple[EventAnalyzer, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    data_validation: Literal["preflight", "stream"] = "preflight"

    def __post_init__(self) -> None:
        if self.snapshot_frequency not in {"bar", "session", "final", "none"}:
            raise ValueError("snapshot_frequency must be one of: bar, session, final, none")
        if self.result_retention not in {"full", "summary"}:
            raise ValueError("result_retention must be 'full' or 'summary'")
        if self.data_validation not in {"preflight", "stream"}:
            raise ValueError("data_validation must be preflight or stream")
        object.__setattr__(self, "analyzers", tuple(self.analyzers))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True, slots=True)
class StrategySpec:
    """A strategy factory with a read-only, shallow-copied parameter mapping.

    Each create() calls the factory, which should return a fresh strategy for
    independent runs. Nested parameter values remain shared.
    """

    factory: StrategyFactory
    parameters: Mapping[str, Any] = field(default_factory=dict)
    name: str | None = None

    def __post_init__(self) -> None:
        if not callable(self.factory):
            raise TypeError("factory must be callable")
        object.__setattr__(self, "parameters", MappingProxyType(dict(self.parameters)))

    @property
    def identity(self) -> str:
        if self.name is not None:
            return self.name
        module = getattr(self.factory, "__module__", type(self.factory).__module__)
        qualname = getattr(self.factory, "__qualname__", type(self.factory).__qualname__)
        return f"{module}.{qualname}"

    def create(self) -> Any:
        return self.factory(**self.parameters)


def run_backtest(
    bars: Iterable[Bar] | ColumnarBarFeed,
    strategy: Strategy | StrategySpec,
    *,
    config: BacktestConfig | None = None,
    metadata: Mapping[str, Any] | None = None,
    assume_sorted: bool = False,
) -> BacktestResult:
    """Run an event backtest using the stable v1 configuration surface.

    Per-run `metadata` is merged into `config.metadata` and stored in the
    result's run manifest. It must contain JSON-compatible values.
    """
    settings = config or BacktestConfig()
    run_metadata = dict(settings.metadata)
    if isinstance(strategy, StrategySpec):
        run_metadata["strategy_spec"] = {
            "identity": strategy.identity,
            "parameters": dict(strategy.parameters),
        }
        strategy = strategy.create()
    if metadata is not None:
        run_metadata.update(metadata)
    engine = BacktestEngine(
        broker=settings.broker.create_broker(),
        trade_on_close=settings.trade_on_close,
        analyzers=list(settings.analyzers),
        snapshot_frequency=settings.snapshot_frequency,
        result_retention=settings.result_retention,
        build_manifest=settings.build_manifest,
        data_validation=settings.data_validation,
        run_metadata={"api_version": API_VERSION, **run_metadata},
    )
    return engine.run(bars, strategy, assume_sorted=assume_sorted)
