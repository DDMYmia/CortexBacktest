"""Pluggable indicator providers supporting native Polars, Tulip Indicators, and pandas-ta."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable
import math
import polars as pl


@runtime_checkable
class IndicatorProvider(Protocol):
    """Protocol for pluggable indicator calculation providers."""

    @property
    def provider_name(self) -> str:
        """Identifier for the provider, e.g. 'builtin', 'tulip', 'pandas_ta'."""
        ...

    def is_available(self) -> bool:
        """Whether this provider's external dependencies are installed."""
        ...

    def list_indicators(self) -> list[str]:
        """List of indicator names supported by this provider."""
        ...

    def compute(self, name: str, data: pl.DataFrame, **kwargs: Any) -> pl.DataFrame:
        """Compute indicator values on a Polars DataFrame (wide matrix or OHLCV table)."""
        ...


class BuiltinPolarsProvider:
    """Built-in provider using native vectorized Polars expressions."""

    def __init__(self, functions: dict[str, Any] | None = None) -> None:
        self._functions = dict(functions or {})

    @property
    def provider_name(self) -> str:
        return "builtin"

    def is_available(self) -> bool:
        return True

    def register(self, name: str, func: Any) -> None:
        self._functions[name.strip().lower().replace("-", "_")] = func

    def list_indicators(self) -> list[str]:
        return sorted(self._functions)

    def compute(self, name: str, data: pl.DataFrame, **kwargs: Any) -> pl.DataFrame:
        key = name.strip().lower().replace("-", "_")
        if key not in self._functions:
            raise KeyError(f"Built-in indicator '{name}' not found")
        return self._functions[key](data, **kwargs)


class TulipIndicatorProvider:
    """Pluggable adapter for Tulip Indicators (tulipindicators / tulipy).

    Tulip Indicators is written in pure ANSI C for extreme performance.
    """

    def __init__(self) -> None:
        self._tulipy: Any = None
        self._checked = False

    @property
    def provider_name(self) -> str:
        return "tulip"

    def is_available(self) -> bool:
        if not self._checked:
            try:
                import tulipy
                self._tulipy = tulipy
            except ModuleNotFoundError as error:
                if error.name != "tulipy":
                    raise
                self._tulipy = None
            self._checked = True
        return self._tulipy is not None

    def list_indicators(self) -> list[str]:
        if not self.is_available():
            return []
        return [
            attr
            for attr in dir(self._tulipy)
            if not attr.startswith("_") and callable(getattr(self._tulipy, attr, None))
        ]

    def requirements(self, name: str, **params: Any) -> dict:
        return external_requirements(self.provider_name, name, params)

    def compute(self, name: str, data: pl.DataFrame, **kwargs: Any) -> pl.DataFrame:
        return _external_compute(self, name, data, kwargs)

    def _compute(self, name: str, data: pl.DataFrame, **kwargs: Any) -> pl.DataFrame:
        if not self.is_available():
            raise ImportError(
                f"Tulip indicator '{name}' requested but 'tulipy' is not installed. "
                "Install with 'pip install tulipy'."
            )
        tulip_fn = getattr(self._tulipy, name.lower(), None)
        if tulip_fn is None:
            raise KeyError(f"Unknown Tulip indicator: {name}")

        # Check if data is a wide price matrix (date + symbols) or an OHLCV table
        columns = [c for c in data.columns if c != "date"]
        has_ohlc = {"high", "low", "close"}.issubset(columns)

        import numpy as np

        if has_ohlc:
            # Single-asset OHLCV table mode
            # tulipy declares each indicator's inputs (`sma` -> ["real"],
            # `atr` -> ["high", "low", "close"]); "real" is the close. Calling
            # one signature and retrying another on any exception turned a
            # genuine computation error into a silent change of inputs.
            columns_by_input = {"real": "close", "open": "open", "high": "high", "low": "low",
                                "close": "close", "volume": "volume"}
            try:
                inputs = [
                    np.ascontiguousarray(data[columns_by_input[item]].to_numpy(), dtype=np.float64)
                    for item in tulip_fn.inputs
                ]
            except KeyError as exc:
                raise ValueError(f"Tulip indicator '{name}' needs input {exc} this table does not have") from exc
            raw_res = tulip_fn(*inputs, **kwargs)

            if isinstance(raw_res, tuple):
                res_dict: dict[str, Any] = {"date": data["date"]}
                for idx, arr in enumerate(raw_res):
                    pad_len = len(data) - len(arr)
                    padded = np.pad(arr, (pad_len, 0), constant_values=np.nan)
                    res_dict[f"{name}_{idx}"] = padded
                return pl.DataFrame(res_dict)
            else:
                pad_len = len(data) - len(raw_res)
                padded = np.pad(raw_res, (pad_len, 0), constant_values=np.nan)
                return pl.DataFrame({"date": data["date"], name: padded})
        else:
            # Cross-sectional wide matrix mode (date x symbols)
            result_series: dict[str, Any] = {"date": data["date"]}
            for col in columns:
                arr_in = np.ascontiguousarray(data[col].to_numpy(), dtype=np.float64)
                raw = tulip_fn(arr_in, **kwargs)
                if isinstance(raw, tuple):
                    raw = raw[0]  # Take primary line for wide matrix
                pad_len = len(data) - len(raw)
                result_series[col] = np.pad(raw, (pad_len, 0), constant_values=np.nan)
            return pl.DataFrame(result_series)


class PandasTAIndicatorProvider:
    """Pluggable adapter for pandas-ta (or pandas-ta-classic).

    Provides access to 130+ technical analysis indicators.
    """

    def __init__(self) -> None:
        self._pta: Any = None
        self._checked = False

    @property
    def provider_name(self) -> str:
        return "pandas_ta"

    def is_available(self) -> bool:
        if not self._checked:
            try:
                import pandas_ta as pta
                self._pta = pta
            except ModuleNotFoundError as error:
                if error.name != "pandas_ta":
                    raise
                try:
                    import pandas_ta_classic as pta
                    self._pta = pta
                except ModuleNotFoundError as error:
                    if error.name != "pandas_ta_classic":
                        raise
                    self._pta = None
            self._checked = True
        return self._pta is not None

    def list_indicators(self) -> list[str]:
        if not self.is_available():
            return []
        return [cat for cat in dir(self._pta) if not cat.startswith("_") and callable(getattr(self._pta, cat))]

    def requirements(self, name: str, **params: Any) -> dict:
        return external_requirements(self.provider_name, name, params)

    def compute(self, name: str, data: pl.DataFrame, **kwargs: Any) -> pl.DataFrame:
        return _external_compute(self, name, data, kwargs)

    def _compute(self, name: str, data: pl.DataFrame, **kwargs: Any) -> pl.DataFrame:
        if not self.is_available():
            raise ImportError(
                f"pandas-ta indicator '{name}' requested but 'pandas-ta' (or 'pandas-ta-classic') is not installed. "
                "Install with 'pip install pandas-ta-classic'."
            )
        kwargs = {**kwargs, "talib": False}
        pta_fn = getattr(self._pta, name.lower(), None)
        if pta_fn is None:
            raise KeyError(f"Unknown pandas-ta indicator: {name}")

        # Convert Polars DataFrame to pandas DataFrame for computation
        pdf = data.to_pandas()
        columns = [c for c in data.columns if c != "date"]
        has_ohlc = {"high", "low", "close"}.issubset(columns)

        if has_ohlc:
            res = pta_fn(
                close=pdf["close"],
                high=pdf.get("high"),
                low=pdf.get("low"),
                open=pdf.get("open"),
                volume=pdf.get("volume"),
                **kwargs,
            )
            res_df = res.to_frame() if hasattr(res, "to_frame") else res
            res_df["date"] = pdf["date"]
            return pl.from_pandas(res_df)
        else:
            # Wide matrix mode
            import pandas as pd
            res_cols = {"date": pdf["date"]}
            for col in columns:
                res_series = pta_fn(close=pdf[col], **kwargs)
                if isinstance(res_series, pd.DataFrame):
                    res_series = res_series.iloc[:, 0]
                res_cols[col] = res_series
            return pl.from_pandas(pd.DataFrame(res_cols))


# Closed, reviewed metadata: unknown functions/options are not guessed from names.
# RSI needs an extra close; Tulip ATR includes high-low as its first true range.
_EXTERNAL = {"sma": (20, 0), "ema": (20, 0), "rsi": (14, 1), "atr": (14, 0)}


def external_requirements(provider: str, name: str, params: dict) -> dict:
    key = name.strip().lower().replace("-", "_")
    parameter = "period" if provider == "tulip" else "length"
    result = {"name": f"{provider}:{key}", "provider": provider,
        "dependencies": ["tulipy"] if provider == "tulip" else ["pandas-ta or pandas-ta-classic", "pandas"],
        "requirements_known": key in _EXTERNAL, "markets": [], "frequencies": [],
        "scope": "supplied_numeric_rows_no_calendar_or_vendor_coverage_inference"}
    if key not in _EXTERNAL:
        return {**result, "reason": "provider_requirements_unknown", "warmup_bars": None}
    if set(params) - {parameter}:
        raise ValueError("unreviewed provider options; offsets/fill/lookahead/backend overrides are not accepted")
    default, offset = _EXTERNAL[key]
    value = params.get(parameter, default)
    if type(value) is not int or value < 2:
        raise ValueError(f"{parameter} must be an integer >= 2")
    if provider == "pandas_ta" and key == "atr":
        offset = 1
    return {**result, "required_columns": ["high", "low", "close"] if key == "atr" else ["close"],
        "extra_columns": [], "input_kind": "table" if key == "atr" else "prices",
        "warmup_bars": value + offset, "parameters": {parameter: value},
        "backend_policy": "native_tulip" if provider == "tulip" else "talib_false_no_backend_switch",
        "gap_policy": "recompute_each_complete_contiguous_segment"}


def assess_external(provider, name: str, data: pl.DataFrame, params: dict) -> tuple[dict, dict[str, pl.Series]]:
    requirements = provider.requirements(name, **params)
    reasons, masks = [], {}
    available = provider.is_available()
    if not available:
        reasons.append("provider_dependency_unavailable")
    if not requirements["requirements_known"]:
        reasons.append("provider_requirements_unknown")
    elif "date" not in data.columns:
        reasons.append("missing_column:date")
    elif not data.height or data["date"].null_count() or data["date"].n_unique() != data.height:
        reasons.append("invalid_or_duplicate_dates")
    else:
        table = requirements["input_kind"] == "table" or {"high", "low", "close"}.issubset(data.columns)
        columns = requirements["required_columns"] if table else [c for c in data.columns if c != "date"]
        if not columns:
            reasons.append("missing_price_columns")
        for column in columns:
            if column not in data.columns:
                reasons.append(f"missing_column:{column}")
            elif not data.schema[column].is_numeric():
                reasons.append(f"nonnumeric_column:{column}")
        if not reasons or reasons == ["provider_dependency_unavailable"]:
            ordered = data.sort("date")
            valid = {c: ordered[c].is_not_null() & ordered[c].is_finite().fill_null(False) for c in columns}
            if table:
                mask = pl.Series([True] * data.height, dtype=pl.Boolean)
                for value in valid.values():
                    mask = mask & value
                valid = {"table": mask}
            masks = valid
    bars = requirements["warmup_bars"]
    usable = {column: int(((mask.cast(pl.UInt32).rolling_sum(window_size=bars, min_samples=bars) == bars).fill_null(False)).sum())
              for column, mask in masks.items()} if bars else {}
    if not reasons and not any(usable.values()):
        reasons.append("insufficient_complete_history")
    return {"indicator": requirements["name"], "ready": not reasons, "reasons": reasons,
        "warmup_bars": bars, "usable_rows": usable, "requirements": requirements,
        "provider_available": available}, masks


def _external_compute(provider, name: str, data: pl.DataFrame, params: dict) -> pl.DataFrame:
    assessment, masks = assess_external(provider, name, data, params)
    if not assessment["ready"]:
        if "provider_dependency_unavailable" in assessment["reasons"]:
            raise ImportError(f"{provider.provider_name}: {assessment['requirements']['dependencies']} unavailable")
        raise ValueError(f"indicator {assessment['indicator']}: {assessment['reasons']}")
    ordered = data.sort("date")
    bars = assessment["warmup_bars"]
    result = {"date": ordered["date"]}
    for key, mask in masks.items():
        valid = mask.to_list()
        columns = {}
        start = 0
        while start < ordered.height:
            if not valid[start]:
                start += 1
                continue
            end = start + 1
            while end < ordered.height and valid[end]:
                end += 1
            if end - start >= bars:
                piece = ordered.slice(start, end-start)
                if key != "table":
                    piece = piece.select("date", key)
                computed = provider._compute(name.strip().lower().replace("-", "_"), piece, **assessment["requirements"]["parameters"])
                if computed.height != piece.height or not computed["date"].equals(piece["date"]):
                    raise ValueError("provider output must preserve the input decision clock")
                for col in computed.columns:
                    if col == "date":
                        continue
                    values = columns.setdefault(col, [None] * ordered.height)
                    for i, value in enumerate(computed[col].to_list()):
                        if i >= bars-1 and value is not None and math.isfinite(value):
                            values[start+i] = value
            start = end
        if not columns:
            # A short wide symbol stays null while another symbol can be ready.
            if key == "table":
                raise ValueError("insufficient_complete_history")
            columns[key] = [None] * ordered.height
        result.update({col: pl.Series(col, values, dtype=pl.Float64) for col, values in columns.items()})
    return pl.DataFrame(result)
