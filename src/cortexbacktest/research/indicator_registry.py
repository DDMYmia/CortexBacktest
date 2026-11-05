"""Named indicator registry supporting pluggable providers (native Polars, Tulip, pandas-ta)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from inspect import signature

import polars as pl
from typing import Any


from .indicator_providers import (
    BuiltinPolarsProvider,
    IndicatorProvider,
    PandasTAIndicatorProvider,
    TulipIndicatorProvider,
    assess_external,
)
from .indicators import (
    average_true_range,
    bollinger_band_width,
    ema_ratio,
    macd,
    momentum,
    rolling_volatility,
    rolling_zscore,
    rsi,
    sma_ratio,
    stochastic_oscillator,
)

IndicatorFunc = Callable[..., Any]


@dataclass(frozen=True, slots=True)
class IndicatorSpec:
    name: str
    func: IndicatorFunc
    description: str = ""
    provider: str = "builtin"
    required_columns: tuple[str, ...] = ()
    extra_columns: tuple[str, ...] = ()
    input_kind: str = "table"
    min_warmup_bars: int = 0
    warmup_parameter: str | None = None
    warmup_offset: int = 0
    markets: tuple[str, ...] = ()
    frequencies: tuple[str, ...] = ()
    output_unit: str = "unspecified"

    def __post_init__(self) -> None:
        if self.input_kind not in {"prices", "table"}:
            raise ValueError("input_kind must be prices or table")
        for value in (self.min_warmup_bars, self.warmup_offset):
            if type(value) is not int or value < 0:
                raise ValueError("warmup bars and offset must be nonnegative integers")

    def warmup_bars(self, params: dict[str, Any]) -> int:
        if self.warmup_parameter is None:
            return self.min_warmup_bars
        bound = signature(self.func).bind_partial(None, **params)
        bound.apply_defaults()
        value = bound.arguments[self.warmup_parameter]
        if type(value) is not int or value < 1:
            raise ValueError(f"{self.warmup_parameter} must be a positive integer")
        return max(self.min_warmup_bars, value + self.warmup_offset)


class IndicatorRegistry:
    """Registry for reusable research indicators with pluggable provider backends."""

    def __init__(self, specs: list[IndicatorSpec] | None = None) -> None:
        self._specs: dict[str, IndicatorSpec] = {}
        self._providers: dict[str, IndicatorProvider] = {
            "builtin": BuiltinPolarsProvider(),
            "tulip": TulipIndicatorProvider(),
            "pandas_ta": PandasTAIndicatorProvider(),
        }
        for spec in specs or []:
            self.register(spec.name, spec.func, **{k: v for k, v in asdict(spec).items() if k not in {"name", "func"}})

    def get_provider(self, name: str) -> IndicatorProvider:
        """Get an indicator provider by name."""
        if name not in self._providers:
            raise KeyError(f"Unknown indicator provider: {name}")
        return self._providers[name]

    def available_providers(self) -> list[str]:
        """Return list of provider names that are currently available/installed."""
        return [name for name, p in self._providers.items() if p.is_available()]

    def register(
        self,
        name: str,
        func: IndicatorFunc,
        *,
        description: str = "",
        provider: str = "builtin",
        required_columns: tuple[str, ...] = (),
        extra_columns: tuple[str, ...] = (),
        input_kind: str = "table",
        min_warmup_bars: int = 0,
        warmup_parameter: str | None = None,
        warmup_offset: int = 0,
        markets: tuple[str, ...] = (),
        frequencies: tuple[str, ...] = (),
        output_unit: str = "unspecified",
    ) -> None:
        key = _normalize_name(name)
        if key in self._specs:
            raise ValueError(f"indicator already registered: {name}")
        self._specs[key] = IndicatorSpec(
            key, func, description, provider, required_columns, extra_columns, input_kind,
            min_warmup_bars, warmup_parameter, warmup_offset, markets, frequencies, output_unit,
        )
        builtin = self._providers.get("builtin")
        if isinstance(builtin, BuiltinPolarsProvider):
            builtin.register(key, func)

    def get(self, name: str) -> IndicatorFunc:
        return self.spec(name).func

    def run(self, name: str, data: Any, *, input_context: dict[str, str] | None = None, **kwargs: Any) -> Any:
        """Run an indicator by name.

        Supports namespaces like 'tulip:rsi' or 'pandas_ta:vwap'.
        If no namespace is given, searches registered specs first, then checks providers.
        """
        if ":" in name:
            provider_prefix, ind_name = name.split(":", 1)
            provider = self.get_provider(provider_prefix)
            if provider_prefix == "builtin":
                return self.run(ind_name, data, input_context=input_context, **kwargs)
            return provider.compute(ind_name, data, **kwargs)

        key = _normalize_name(name)
        if key in self._specs:
            spec = self.spec(key)
            if not (spec.required_columns or spec.extra_columns or spec.min_warmup_bars
                    or spec.warmup_parameter or spec.markets or spec.frequencies):
                return spec.func(data, **kwargs)
            assessment = self.assess(key, data, input_context=input_context, **kwargs)
            if not assessment["ready"]:
                raise ValueError(f"indicator {key}: {assessment['reasons']}")
            ordered = data.sort("date")
            result = spec.func(ordered, **kwargs)
            masks, _ = self._input_masks(spec, ordered, kwargs)
            expressions = []
            for column in result.columns:
                if column == "date":
                    continue
                mask = masks[column] if spec.input_kind == "prices" else masks["table"]
                expressions.append(pl.when(mask).then(pl.col(column)).otherwise(None).alias(column))
            return result.with_columns(expressions)

        # Fallback: check if any provider supports it
        for prov_name in ["builtin", "tulip", "pandas_ta"]:
            prov = self._providers.get(prov_name)
            if prov and prov.is_available() and key in [i.lower() for i in prov.list_indicators()]:
                return self.run(f"{prov_name}:{key}", data, input_context=input_context, **kwargs)

        raise KeyError(f"unknown indicator: {name}")

    def spec(self, name: str) -> IndicatorSpec:
        key = _normalize_name(name)
        try:
            return self._specs[key]
        except KeyError as exc:
            raise KeyError(f"unknown indicator: {name}") from exc

    def describe(self, name: str, **params: Any) -> dict[str, Any]:
        # Serializable requirements; bars include the first usable output row.
        if ":" in name:
            prefix, key = name.split(":", 1)
            if prefix != "builtin":
                provider = self.get_provider(prefix)
                return getattr(provider, "requirements")(key, **params)
            name = key
        spec = self.spec(name)
        bound = signature(spec.func).bind_partial(None, **params)
        bound.apply_defaults()
        effective = dict(bound.arguments)
        effective.pop(next(iter(signature(spec.func).parameters)), None)
        return {**{k: v for k, v in asdict(spec).items() if k != "func"},
                "warmup_bars": spec.warmup_bars(params),
                "missing_policy": ("null_until_complete_finite_input_window" if spec.required_columns or spec.extra_columns
                    or spec.min_warmup_bars or spec.warmup_parameter else "provider_defined"),
                "effective_parameters": effective}

    def assess(
        self, name: str, data: pl.DataFrame, *,
        input_context: dict[str, str] | None = None, **params: Any,
    ) -> dict[str, Any]:
        # Inspect supplied rows only; infer no calendar or source availability.
        if ":" in name:
            prefix, key = name.split(":", 1)
            if prefix != "builtin":
                return assess_external(self.get_provider(prefix), key, data, params)[0]
            name = key
        spec = self.spec(name)
        reasons = []
        context = input_context or {}
        for field, allowed in (("market", spec.markets), ("frequency", spec.frequencies)):
            if allowed and context.get(field) not in allowed:
                reasons.append(f"unsupported_or_missing_{field}")
        if "date" not in data.columns:
            reasons.append("missing_column:date")
            masks = {}
        elif data["date"].null_count() or data["date"].n_unique() != data.height:
            reasons.append("invalid_or_duplicate_dates")
            masks = {}
        else:
            masks, input_reasons = self._input_masks(spec, data.sort("date"), params)
            reasons.extend(input_reasons)
        usable = {column: int(mask.sum()) for column, mask in masks.items()}
        if not reasons and not any(usable.values()):
            reasons.append("insufficient_complete_history")
        return {"indicator": spec.name, "ready": not reasons, "reasons": reasons,
                "warmup_bars": spec.warmup_bars(params), "usable_rows": usable,
                "requirements": self.describe(name, **params)}

    def _input_masks(
        self, spec: IndicatorSpec, data: pl.DataFrame, params: dict[str, Any],
    ) -> tuple[dict[str, pl.Series], list[str]]:
        columns = ([c for c in data.columns if c != "date" and c not in spec.extra_columns]
                   if spec.input_kind == "prices" else
                   [params.get(c + "_col", c) for c in spec.required_columns])
        missing = [c for c in [*columns, *spec.extra_columns] if c not in data.columns]
        if missing:
            return {}, [f"missing_column:{c}" for c in missing]
        if not columns:
            return {}, ["missing_price_columns"]
        invalid = [c for c in columns if not data.schema[c].is_numeric()]
        if invalid:
            return {}, [f"nonnumeric_column:{c}" for c in invalid]
        valid = {c: data[c].is_not_null() & data[c].is_finite().fill_null(False) for c in columns}
        extra = pl.Series([True] * data.height, dtype=pl.Boolean)
        for column in spec.extra_columns:
            extra = extra & data[column].is_not_null()
        if spec.input_kind == "table":
            for mask in valid.values():
                extra = extra & mask
            valid = {"table": extra}
        bars = max(1, spec.warmup_bars(params))
        return {c: ((mask & extra).cast(pl.UInt32).rolling_sum(
            window_size=bars, min_samples=bars) == bars).fill_null(False)
                for c, mask in valid.items()}, []

    def names(self) -> list[str]:
        return sorted(self._specs)


def default_indicator_registry() -> IndicatorRegistry:
    return IndicatorRegistry(
        [
            IndicatorSpec("momentum", momentum, "Price momentum over a lookback window.", required_columns=("close",), input_kind="prices", warmup_parameter="window", warmup_offset=1, output_unit="fractional_return"),
            IndicatorSpec("sma_ratio", sma_ratio, "Price divided by simple moving average minus one.", required_columns=("close",), input_kind="prices", warmup_parameter="window", warmup_offset=0, output_unit="dimensionless_ratio_minus_one"),
            IndicatorSpec("ema_ratio", ema_ratio, "Price divided by exponential moving average minus one.", required_columns=("close",), input_kind="prices", warmup_parameter="span", warmup_offset=0, output_unit="dimensionless_ratio_minus_one"),
            IndicatorSpec("rolling_volatility", rolling_volatility, "Rolling close-to-close realized volatility.", required_columns=("close",), input_kind="prices", warmup_parameter="window", warmup_offset=1, output_unit="fractional_return_std_times_annualization"),
            IndicatorSpec("rsi", rsi, "Relative strength index scaled to 0-100.", required_columns=("close",), input_kind="prices", warmup_parameter="window", warmup_offset=1, output_unit="points_0_to_100"),
            IndicatorSpec("macd", macd, "Moving average convergence/divergence line.", required_columns=("close",), input_kind="prices", warmup_parameter="slow", warmup_offset=0, output_unit="input_price_unit"),
            IndicatorSpec("bollinger_band_width", bollinger_band_width, "Normalized Bollinger band width.", required_columns=("close",), input_kind="prices", warmup_parameter="window", warmup_offset=0, output_unit="dimensionless_ratio"),
            IndicatorSpec("rolling_zscore", rolling_zscore, "Rolling price z-score.", required_columns=("close",), input_kind="prices", warmup_parameter="window", warmup_offset=0, output_unit="standard_deviations"),
            IndicatorSpec("average_true_range", average_true_range, "Average true range for OHLC bars.", required_columns=("high", "low", "close"), input_kind="table", warmup_parameter="window", warmup_offset=0, output_unit="input_price_unit"),
            IndicatorSpec("stochastic_oscillator", stochastic_oscillator, "Stochastic oscillator %K.", required_columns=("high", "low", "close"), input_kind="table", warmup_parameter="window", warmup_offset=0, output_unit="points_0_to_100"),
        ]
    )


def _normalize_name(name: str) -> str:
    key = name.strip().lower().replace("-", "_")
    if not key:
        raise ValueError("indicator name must not be empty")
    return key


INDICATORS = default_indicator_registry()
