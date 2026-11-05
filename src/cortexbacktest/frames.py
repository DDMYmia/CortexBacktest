"""Stable import path for research frames.

Re-exports .research.frames so callers can import by concept rather than by the
sub-package that currently owns the code. Pinned by
`tests/test_public_module_paths.py`; see that module for why this layer
exists and what it guarantees.
"""

from .research.frames import *  # noqa: F403
