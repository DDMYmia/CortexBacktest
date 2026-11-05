"""Dynamic strategy loading helpers for operational entrypoints."""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import json
from pathlib import Path
from types import ModuleType
from typing import Any

from .strategy import Strategy


def load_strategy(
    spec: str,
    *,
    kwargs: dict[str, Any] | None = None,
    broker: Any = None,
    dataset: Any = None,
) -> Strategy:
    """Load and instantiate a Strategy from `module:attribute` or `file.py:attribute`."""
    module_name, attribute = _split_strategy_spec(spec)
    module = _load_module(module_name)
    try:
        target = getattr(module, attribute)
    except AttributeError as exc:
        raise ValueError(f"Strategy attribute not found: {attribute}") from exc
    strategy = _instantiate_strategy(target, kwargs=kwargs or {}, broker=broker, dataset=dataset)
    if not isinstance(strategy, Strategy):
        raise TypeError("Loaded strategy must be an instance of cortexbacktest.Strategy")
    return strategy


def parse_strategy_kwargs(value: str | None) -> dict[str, Any]:
    """Parse JSON object kwargs passed to a dynamic strategy."""
    if value is None or value.strip() == "":
        return {}
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("strategy kwargs must be a JSON object")
    return parsed


def _split_strategy_spec(spec: str) -> tuple[str, str]:
    if ":" not in spec:
        raise ValueError("strategy spec must use 'module:attribute' or 'path.py:attribute'")
    module_name, attribute = spec.rsplit(":", 1)
    if not module_name or not attribute:
        raise ValueError("strategy spec must include both module/path and attribute")
    return module_name, attribute


def _load_module(module_name: str) -> ModuleType:
    path = Path(module_name).expanduser()
    if path.suffix == ".py" or path.exists():
        if not path.exists():
            raise FileNotFoundError(f"Strategy file does not exist: {path}")
        spec = importlib.util.spec_from_file_location(f"_fusion_strategy_{path.stem}", path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load strategy file: {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    return importlib.import_module(module_name)


def _instantiate_strategy(
    target: Any,
    *,
    kwargs: dict[str, Any],
    broker: Any,
    dataset: Any,
) -> Strategy:
    if isinstance(target, Strategy):
        if kwargs:
            raise ValueError("strategy kwargs cannot be supplied when loading a strategy instance")
        return target
    if not callable(target):
        raise TypeError("strategy attribute must be a Strategy instance, class, or factory")
    call_kwargs = dict(kwargs)
    signature = inspect.signature(target)
    for name, value in (("broker", broker), ("dataset", dataset)):
        if name in signature.parameters and name not in call_kwargs:
            call_kwargs[name] = value
    strategy = target(**call_kwargs)
    if not isinstance(strategy, Strategy):
        raise TypeError(f"strategy factory must return a Strategy, got {type(strategy).__qualname__}")
    return strategy
