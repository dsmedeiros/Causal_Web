"""Bell experiment helpers for the v2 engine.

This module contains placeholders for ancestry tracking and contextual
readouts used in Bell-type experiments. The implementations are
non-functional and serve only to define the expected interface.
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class Ancestry:
    """Track simple ancestry information for a packet."""

    parent_id: int | None = None


class BellHelpers:
    """Utility methods for Bell experiment simulations."""

    def __init__(self, seed: int | None = None) -> None:
        self._rand = random.Random(seed)

    def lambda_at_source(self) -> float:
        """Return a random ``lambda`` value."""

        return self._rand.random()

    def setting_draw(self) -> int:
        """Draw a binary measurement setting."""

        return self._rand.randint(0, 1)

    def contextual_readout(self, setting: int, lam: float) -> int:
        """Produce a contextual readout given ``setting`` and ``lambda``."""

        return 1 if lam > 0.5 else -1
