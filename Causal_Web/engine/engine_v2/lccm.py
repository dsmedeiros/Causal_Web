"""Local causal consistency model helpers.

This module exposes placeholder functions for Q accumulation and
window management used by the experimental engine. The routines are
simple stubs that will be replaced with full implementations as the
physics model evolves.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LCCM:
    """Accumulate and mix local causal consistency metrics."""

    q: float = 0.0

    def accumulate(self, value: float) -> None:
        """Accumulate ``value`` into ``q``."""

        self.q += value

    def close(self) -> float:
        """Return the accumulated ``q`` and reset it."""

        result = self.q
        self.q = 0.0
        return result

    def majority(self, a: float, b: float, c: float) -> float:
        """Simple majority helper used in windowing math."""

        return (a + b + c) / 3.0
