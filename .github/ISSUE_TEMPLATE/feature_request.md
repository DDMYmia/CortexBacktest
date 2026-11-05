---
name: Feature request
about: Propose a capability for CortexBacktest
title: "[Feature]: "
labels: enhancement
assignees: ""
---

## Problem

<!-- the research or backtesting question you cannot answer today -->

## Proposed Capability


## Prior Art

<!-- how zipline / backtrader / vectorbt / alphalens handle this, if they do -->

## Correctness Considerations

<!-- what would have to be true for results using this feature to be trustworthy:
     look-ahead bias, survivorship, timing of data availability, cost modelling -->

## Scope Check

- [ ] Falls inside the engine boundary in `docs/decisions.md` (decision 2)
      (if it is data warehousing, PIT fundamental storage, experiment
      management, live routing, or UI, it belongs to a sibling project)
- [ ] Fits the Polars-only core (or is confined to the optional data boundary)
- [ ] Runtime dependency changes are declared in `pyproject.toml`
