"""Free-energy score for Θ→C transitions."""

from __future__ import annotations

from typing import Dict


def free_energy_score(
    entropy: float,
    confidence: float,
    eq_recent: float,
    *,
    k_theta: float,
    k_c: float,
    k_q: float,
) -> float:
    """Return the free-energy score ``F_v`` for a vertex."""

    return k_theta * (1.0 - entropy) + k_c * confidence - k_q * eq_recent


def stamp_lccm_metadata(
    meta: Dict[str, object], mode: str, params: Dict[str, object]
) -> None:
    """Annotate run metadata with the LCCM mode and parameters."""

    meta["lccm"] = {"mode": mode, **params}


__all__ = ["free_energy_score", "stamp_lccm_metadata"]
