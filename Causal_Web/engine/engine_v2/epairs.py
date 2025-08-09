"""Epsilon-pair management for the v2 engine.

The routines below track time-to-live (TTL) seeds and provide a simple
reinforcement/decay mechanism for bridge ``sigma`` values. They are
non-physical placeholders that allow higher layers to exercise the API
without requiring a completed physics model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class TTLSeeds:
    """Track TTL values for epsilon pair seeds."""

    seeds: Dict[int, int] = field(default_factory=dict)

    def seed(self, pair_id: int, ttl: int) -> None:
        self.seeds[pair_id] = ttl

    def tick(self) -> None:
        expired = [k for k, v in self.seeds.items() if v <= 1]
        for k in expired:
            del self.seeds[k]
        for k in self.seeds:
            self.seeds[k] -= 1


@dataclass
class BridgeSigma:
    """Reinforce or decay a bridge's sigma value."""

    sigma: float = 0.0
    decay: float = 0.1

    def reinforce(self, amount: float) -> None:
        self.sigma += amount

    def tick(self) -> None:
        self.sigma *= 1.0 - self.decay
