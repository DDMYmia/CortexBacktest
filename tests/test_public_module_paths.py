"""Pin supported flat import paths and explicit typed export catalogues.

Flat facades must expose every public implementation name. Catalogue and
canonical-import inventory checks separately require exact sets without duplicates.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest


def _is_star_reexport(path: Path) -> bool:
    """Whether a module's entire body is one `from ... import *`.

    Defined locally rather than imported from `test_architecture_boundaries`:
    importing across test modules breaks when pytest is invoked from outside
    the repository root, which CI does not do but a developer easily might.
    """
    body = [node for node in ast.parse(path.read_text(encoding="utf-8")).body
            if not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant))]
    return (
        len(body) == 1
        and isinstance(body[0], ast.ImportFrom)
        and any(alias.name == "*" for alias in body[0].names)
    )

# path -> the module it fronts. Kept explicit rather than discovered, so that
# adding a shim without deciding what it fronts fails the test.
PUBLIC_MODULE_PATHS = {
    "broker": "event.broker",
    "costs": "event.costs",
    "engine": "event.engine",
    "frames": "research.frames",
    "rebalance": "event.rebalance",
    "records": "reporting.records",
    "strategy": "event.strategy",
}


def _public_names(module) -> set[str]:
    exported = getattr(module, "__all__", None)
    if exported is not None:
        return set(exported)
    return {name for name in dir(module) if not name.startswith("_")}


@pytest.mark.parametrize("path, target", sorted(PUBLIC_MODULE_PATHS.items()))
def test_flat_path_re_exports_its_target(path, target):
    facade = importlib.import_module(f"cortexbacktest.{path}")
    implementation = importlib.import_module(f"cortexbacktest.{target}")

    facade_names = _public_names(facade)
    implementation_names = _public_names(implementation)

    assert facade_names, f"{path} re-exports nothing"
    missing = implementation_names - facade_names
    assert not missing, f"{path} does not re-export {sorted(missing)} from {target}"


def test_every_flat_path_module_is_declared_here():
    # A new re-export module must be added to the mapping above, so the decision
    # "what does this front, and is it public?" cannot be skipped. Discovery is
    # structural — a body of exactly one `from ... import *` — matching
    # `test_architecture_boundaries._is_star_reexport`.
    import cortexbacktest

    package_dir = Path(next(iter(cortexbacktest.__path__)))
    shims = {source.stem for source in package_dir.glob("*.py") if _is_star_reexport(source)}

    assert shims == set(PUBLIC_MODULE_PATHS), (
        f"undeclared re-export modules: {sorted(shims - set(PUBLIC_MODULE_PATHS))}; "
        f"declared but missing: {sorted(set(PUBLIC_MODULE_PATHS) - shims)}"
    )


def test_canonical_import_inventory_covers_public_names_once():
    import re
    import cortexbacktest as package

    documentation = Path(__file__).resolve().parents[1] / "docs" / "api.md"
    # Resolve declared exports from their source; importing the historical
    # performance_report shim overwrites the same-named package attribute.
    declared = {}
    for node in ast.parse(Path(package.__file__).read_text()).body:
        if isinstance(node, ast.ImportFrom):
            source = importlib.import_module("cortexbacktest." + node.module)
            for alias in node.names:
                declared[alias.asname or alias.name] = getattr(source, alias.name)
    names = []
    for line in documentation.read_text().splitlines():
        if not line.startswith("| `"):
            continue
        module_name, exported = line.split("|")[1:3]
        module = importlib.import_module("cortexbacktest." + module_name.strip().strip("`"))
        for name in re.findall(r"`([^`]+)`", exported):
            assert getattr(module, name) is declared[name]
            names.append(name)
    assert len(names) == len(set(names))
    assert set(names) == set(package.__all__)


@pytest.mark.parametrize("suffix", ["", ".research", ".reporting", ".event", ".data", ".datasets"])
def test_facade_catalogue_is_exactly_its_explicit_typed_reexports(suffix):
    module = importlib.import_module("cortexbacktest" + suffix)
    tree = ast.parse(Path(module.__file__).read_text())
    names = []
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                assert alias.asname == alias.name, "Public bindings must be explicit typed reexports"
                names.append(alias.name)
    assert len(names) == len(set(names))
    assert set(module.__all__) == set(names)
