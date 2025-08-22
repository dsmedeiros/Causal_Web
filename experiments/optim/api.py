from __future__ import annotations

"""Common optimizer interface used by experiment harnesses."""

from typing import Dict, List, Protocol, runtime_checkable


@runtime_checkable
class Optimizer(Protocol):
    """Protocol for optimizers exploring configuration spaces."""

    def suggest(self, n: int) -> List[Dict[str, float]]:
        """Return ``n`` candidate configurations."""

    def observe(self, results: List[Dict[str, float]]) -> None:
        """Report observed fitness for previously suggested configs."""

    def done(self) -> bool:
        """Return ``True`` when the search is complete."""
