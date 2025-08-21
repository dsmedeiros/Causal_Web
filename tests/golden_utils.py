"""Helpers for working with golden replay logs in tests."""

from __future__ import annotations

import json
from typing import Any, Tuple


def expected_from_log(path: str) -> Tuple[int, float]:
    """Return the final frame index and residual EWMA from ``path``.

    The golden logs are line-delimited JSON where the final line holds the
    expected state after replay. Only the ``frame`` index and the
    ``residual_ewma`` invariant are extracted.
    """

    last: dict[str, Any] | None = None
    with open(path) as fh:  # pragma: no cover - simple file scan
        for line in fh:
            last = json.loads(line)
    if last is None:
        raise ValueError(f"log {path} is empty")
    return last.get("frame", 0), last.get("invariants", {}).get("residual_ewma", 0.0)
