"""Stub interpreter for legacy log files."""

# TODO: legacy refactor

from __future__ import annotations
from typing import Any


def run_interpreter(*args: Any, **kwargs: Any) -> None:
    """Placeholder that performs no interpretation.

    The tick-era log interpreter has been moved to ``legacy/``. This stub
    remains so modules depending on it can import without error.
    """

    return None
