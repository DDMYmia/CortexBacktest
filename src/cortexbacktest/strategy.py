"""Stable import path for the event strategy API.

Re-exports .event.strategy so callers can import by concept rather than by the
sub-package that currently owns the code. Pinned by
`tests/test_public_module_paths.py`; see that module for why this layer
exists and what it guarantees.
"""

from .event.strategy import *  # noqa: F403
