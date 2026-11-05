## Summary

- 

## Verification

- [ ] `python -m ruff check src tests`
- [ ] `python -m basedpyright src/cortexbacktest` (0 errors)
- [ ] `python -m pytest -q`
- [ ] Clean-install gate run when packaging, dependencies, or imports changed

## Scope

- [ ] Public API docs updated when the API changed
- [ ] `CHANGELOG.md` updated for user-visible changes
- [ ] Benchmark numbers regenerated when a doc's numbers changed
- [ ] Tests asserting on reference engines guard with `_require_references(...)`
- [ ] Runtime dependency changes are declared in `pyproject.toml`
