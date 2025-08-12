from __future__ import annotations

"""Local curvature diagnostics."""

import logging
from collections import defaultdict
from typing import Dict, Iterable, List

import numpy as np

logger = logging.getLogger(__name__)


def forman_curvature(d_eff: float, deg_u: int, deg_v: int) -> float:
    """Compute a Forman-style curvature for a directed edge.

    Parameters
    ----------
    d_eff:
        Effective delay associated with the edge.
    deg_u:
        Outgoing degree of the source vertex.
    deg_v:
        Incoming degree of the destination vertex.

    Returns
    -------
    float
        Local curvature value ``F_e``.
    """

    if d_eff <= 0:
        raise ValueError("d_eff must be positive")
    term_u = 1.0 / deg_u if deg_u else 0.0
    term_v = 1.0 / deg_v if deg_v else 0.0
    return (term_u + term_v) / d_eff


class CurvatureLogger:
    """Aggregate and emit curvature statistics per region."""

    def __init__(self) -> None:
        self._values: Dict[str, List[float]] = defaultdict(list)

    def log_edge(self, region: str, curvature: float) -> None:
        """Record curvature for an edge belonging to ``region``."""

        self._values[region].append(curvature)

    def window_close(self) -> Dict[str, Dict[str, float]]:
        """Flush accumulated values and return per-region stats.

        The function logs curvature histograms for each region and returns a
        mapping ``region -> {mean, var}``.
        """

        stats: Dict[str, Dict[str, float]] = {}
        for region, vals in self._values.items():
            arr = np.asarray(vals, dtype=float)
            stats[region] = {
                "mean": float(arr.mean()),
                "var": float(arr.var()),
            }
            hist, bin_edges = np.histogram(arr, bins="auto")
            logger.info(
                "curvature_hist",
                extra={
                    "region": region,
                    "hist": hist.tolist(),
                    "bins": bin_edges.tolist(),
                },
            )
        self._values.clear()
        return stats


__all__ = ["forman_curvature", "CurvatureLogger"]
