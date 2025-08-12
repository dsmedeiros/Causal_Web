"""Invariant checks used during DOE runs."""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Sequence, Tuple


def causality(deliveries: Iterable[Mapping[str, float]]) -> bool:
    """Ensure no delivery arrives before its source."""

    return all(d["d_arr"] >= d["d_src"] for d in deliveries)


def local_conservation(prev: float, current: float, tolerance: float) -> bool:
    """Check that local conservation holds within ``tolerance``."""

    return abs(current - prev) <= tolerance


def no_signaling(prob: float, epsilon: float) -> bool:
    """Verify single-site marginals are near 0.5."""

    return abs(prob - 0.5) < epsilon


def ancestry_determinism(seq: Sequence[Tuple[str, str, str]]) -> bool:
    """Identical local Q sequences imply identical (h, m)."""

    seen: dict[Tuple[str, ...], Tuple[str, str]] = {}
    for q, h, m in seq:
        if q in seen and seen[q] != (h, m):
            return False
        seen[q] = (h, m)
    return True


def from_metrics(m: Mapping[str, float | bool]) -> Dict[str, float | bool]:
    """Extract invariant fields from a metrics mapping."""

    return {
        "inv_causality_ok": bool(m.get("inv_causality_ok", True)),
        "inv_conservation_residual": float(m.get("inv_conservation_residual", 0.0)),
        "inv_no_signaling_delta": float(m.get("inv_no_signaling_delta", 0.0)),
        "inv_ancestry_ok": bool(m.get("inv_ancestry_ok", True)),
    }
