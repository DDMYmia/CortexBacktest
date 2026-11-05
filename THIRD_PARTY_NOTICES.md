# Third-Party Notices

CortexBacktest studies these projects:

- Zipline Reloaded: Apache-2.0
- Backtrader: GPL-3.0
- Backtesting.py: AGPL-3.0
- VectorBT: Apache-2.0 with Commons Clause
- Lumibot: GPL-3.0

None of them is a dependency. The benchmark harness in
`benchmarks/real_data_comparison/` installs each compared engine in its own
environment, outside this package.

## Direct Source Reuse

No direct third-party source code has been copied into
`src/cortexbacktest` as of this notice.

If direct source adaptation is introduced later, list:

- reference project and file
- CortexBacktest file
- authorization or license basis
- summary of modifications

## Architecture References

The project architecture is inspired by:

- Zipline Reloaded Pipeline/Blotter/Ledger separation.
- Backtrader orchestrator/analyzer separation.
- Backtesting.py minimal strategy/result API.
- VectorBT matrix and records-first workflows.
- Lumibot broker/data-source/strategy separation.
