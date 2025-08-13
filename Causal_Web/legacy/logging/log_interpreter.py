"""Stub interpreter for legacy log files."""

from __future__ import annotations
import warnings
from typing import Any

warnings.warn(
    "Causal_Web.legacy.logging.log_interpreter is deprecated and will be removed",
    DeprecationWarning,
    stacklevel=2,
)


def run_interpreter(*args: Any, **kwargs: Any) -> None:
    """Placeholder that performs no interpretation.

    The tick-era log interpreter has been moved to ``legacy/``. This stub
    remains so modules depending on it can import without error.
    """

    return None
