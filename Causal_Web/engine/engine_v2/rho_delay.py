"""Density-dependent delay helpers.

Functions here provide a toy diffusion model and a saturating mapping
from density to an effective delay. They are intentionally simple and
serve as placeholders until the full model is implemented.
"""

from __future__ import annotations

from typing import Iterable, List


def diffuse(rho: List[float], weight: float) -> List[float]:
    """Diffuse density values across neighbours.

    Parameters
    ----------
    rho:
        List of density values.
    weight:
        Diffusion weight in ``[0, 1]``.
    """

    if not rho:
        return []
    avg = sum(rho) / len(rho)
    return [r + weight * (avg - r) for r in rho]


def effective_delay(rho: float, cap: float = 1.0) -> float:
    """Map density to an effective delay using a simple saturation."""

    return min(rho, cap)
