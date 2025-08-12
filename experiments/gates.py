"""Gate metric helpers.

This module provides a thin wrapper around the engine's gate
benchmark suite. The implementation here is a placeholder that
should be wired to the real engine entry points.  Each gate
execution returns a flat mapping of metric names to values.
"""

from __future__ import annotations

from typing import Dict, List


def run_gates(config: Dict[str, float], which: List[int]) -> Dict[str, float]:
    """Execute selected gates and collect metrics.

    Parameters
    ----------
    config:
        Engine configuration passed to the gate harness.
    which:
        List of gate identifiers to execute.

    Returns
    -------
    dict
        Mapping of metric names to values.  Invariants derived
        from the gate results may also be included in the mapping.

    Notes
    -----
    This function currently returns placeholder invariant fields and
    per-gate metrics and should be connected to the real engine
    implementation.
    """

    # TODO: wire to your engine entrypoints
    metrics = {f"G{g}": float(g) for g in which}
    metrics.update(
        {
            "inv_causality_ok": True,
            "inv_conservation_residual": 0.0,
            "inv_no_signaling_delta": 0.0,
            "inv_ancestry_ok": True,
        }
    )
    return metrics
