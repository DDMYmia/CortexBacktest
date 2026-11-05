import ast
from pathlib import Path


SRC_ROOT = Path(__file__).resolve().parents[1] / "src" / "cortexbacktest"


def _imports_for(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = "." * node.level + (node.module or "")
            imports.add(module)
            # Include package members: `from .. import event` and
            # `from cortexbacktest import research` also cross the boundary.
            separator = "" if module.endswith(".") else "."
            imports.update(module + separator + alias.name for alias in node.names if alias.name != "*")
    return imports


def _is_star_reexport(path: Path) -> bool:
    """Whether a module's entire body is one `from ... import *`."""
    body = [node for node in ast.parse(path.read_text()).body
            if not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant))]
    return (
        len(body) == 1
        and isinstance(body[0], ast.ImportFrom)
        and any(alias.name == "*" for alias in body[0].names)
    )


def test_core_modules_do_not_import_external_app_or_references():
    forbidden = ("backend", "references")
    for path in SRC_ROOT.rglob("*.py"):
        for imported in _imports_for(path):
            assert not imported.startswith(forbidden), f"{path} imports {imported}"


def test_research_and_event_workflows_do_not_depend_on_each_other():
    research_imports = set()
    for path in (SRC_ROOT / "research").rglob("*.py"):
        research_imports.update(_imports_for(path))

    event_imports = set()
    for path in (SRC_ROOT / "event").rglob("*.py"):
        event_imports.update(_imports_for(path))

    def reaches(imported: str, package: str) -> bool:
        normalized = imported.lstrip(".")
        return any(normalized == prefix or normalized.startswith(prefix + ".")
                   for prefix in (package, "cortexbacktest." + package))

    assert not any(reaches(name, "event") for name in research_imports), research_imports
    assert not any(reaches(name, "research") for name in event_imports), event_imports


def test_compatibility_wrappers_stay_thin():
    """A flat import path must re-export and nothing else.

    Checked structurally rather than by line count: the invariant is "no logic
    here", and a length limit also caps the docstring, which is where these
    modules explain why they exist. The wrappers are discovered rather than
    listed, so this shares one source of truth with
    `test_public_module_paths.py` instead of keeping a second copy of the list
    to drift out of sync.
    """
    wrappers = [path for path in sorted(SRC_ROOT.glob("*.py")) if _is_star_reexport(path)]
    assert len(wrappers) >= 7, f"expected the flat import paths, found {[p.name for p in wrappers]}"

    for path in wrappers:
        body = list(ast.parse(path.read_text()).body)
        if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
            docstring = body.pop(0).value.value
            assert "Stable import path" in docstring, f"{path.name} does not say what it is"
        assert len(body) == 1, f"{path.name} has code beyond its re-export: {ast.dump(body[1])[:80]}"


def test_reference_reuse_policy_is_documented():
    root = SRC_ROOT.parents[1]
    assert "Reusing other engines" in (root / "docs" / "decisions.md").read_text()
    assert (root / "THIRD_PARTY_NOTICES.md").exists()
    # Authorization records stay in the private repository; the public export
    # ships the policy and notices, which state that no source was copied.
    if (root / "docs" / "authorizations").exists():
        assert (root / "docs" / "authorizations" / "README.md").exists()
        assert (root / "docs" / "authorizations" / "reference-projects-authorization.md").exists()


def test_simulation_and_record_sink_do_not_import_reporting():
    for filename in ("loop.py", "result.py", "record_sink.py", "dispatch.py"):
        imports = _imports_for(SRC_ROOT / "event" / filename)
        assert not any("reporting" in name or "runtime" in name for name in imports)


def test_every_process_pool_spawns_instead_of_forking():
    # Linux defaults to fork; a forked child inherits Polars' thread pool
    # mid-state and deadlocks (CI hung 45 minutes on the first Linux run).
    pools = []
    for path in SRC_ROOT.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "ProcessPoolExecutor":
                context = next((k.value for k in node.keywords if k.arg == "mp_context"), None)
                spawns = (isinstance(context, ast.Call) and getattr(context.func, "id", None) == "get_context"
                          and [a.value for a in context.args if isinstance(a, ast.Constant)] == ["spawn"])
                pools.append((path.name, node.lineno, spawns))
    assert pools, "expected at least one process pool"
    assert all(spawns for *_, spawns in pools), [p for p in pools if not p[2]]
