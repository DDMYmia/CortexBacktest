# Security Policy

## Supported versions

CortexBacktest is a 0.x preview. Only the latest `main` and the most
recent `0.1.x` tag receive fixes.

| Version | Supported |
| --- | --- |
| 0.1.x | yes |
| < 0.1 | no |

## Reporting a vulnerability

Report privately — do not open a public issue.

- Use GitHub's **Report a vulnerability** button under the repository's Security
  tab (private vulnerability reporting), or
- email the maintainer listed in [`.github/CODEOWNERS`](.github/CODEOWNERS).

Please include the affected version or commit, reproduction steps, and what an
attacker gains. Expect an acknowledgement within about a week; this is a
single-maintainer project, not a staffed security team.

## Scope

This is a research and backtesting library. It reads market data files and
executes user-supplied strategy code in the caller's own process — a strategy is
ordinary Python and is trusted by definition, so "a strategy can run arbitrary
code" is not a vulnerability here.

Realistic issues include: unsafe deserialization of dataset, manifest, or run
files; path traversal in dataset and report writing; and resource exhaustion
from crafted input data.

Out of scope: the competitor engines installed for `benchmarks/real_data_comparison/`
(not dependencies of the package), and the recorded results in `benchmark-results/`.

## Dependency policy

The runtime dependency set is deliberately small (NumPy, Polars, PyArrow,
orjson); everything else is an optional extra. The `clean-install` CI job
installs the built distribution into an environment holding only the declared
dependencies, so an undeclared runtime dependency cannot slip in unnoticed.
