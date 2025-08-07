"""Stress-energy density field.

Track energy accumulated on edges and allow diffusion across the graph.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Tuple

import numpy as np

from ..models.node import Edge


class DensityField:
    """Maintain a map of density values for edges.

    Density is updated from amplitude energy contributions and diffuses across
    neighbouring edges each scheduler step.
    """

    def __init__(self) -> None:
        self._rho: Dict[Tuple[str, str], float] = defaultdict(float)

    # ------------------------------------------------------------------
    def deposit(self, edge: Edge, amplitude: complex) -> None:
        """Accumulate density on ``edge`` from ``amplitude``.

        Parameters
        ----------
        edge:
            Edge receiving the energy contribution.
        amplitude:
            Complex amplitude whose squared magnitude represents energy.

        Note
        ----
        Access to the internal density map is protected by the GIL in
        CPython but is not inherently thread-safe. Alternative interpreters
        or accelerator backends may require atomic updates or per-thread
        accumulation.
        """
        # TODO: Implement CuPy kernel for energy accumulation
        energy = float(np.sum(np.abs(amplitude) ** 2))
        self._rho[(edge.source, edge.target)] += energy

    # ------------------------------------------------------------------
    def get(self, edge: Edge) -> float:
        """Return current density value for ``edge``."""

        return self._rho.get((edge.source, edge.target), 0.0)

    # ------------------------------------------------------------------
    def diffuse(self, graph: "CausalGraph", alpha: float) -> None:
        """Diffuse density across adjacent edges.

        Parameters
        ----------
        graph:
            Graph providing edge adjacency information.
        alpha:
            Weight for diffusion; ``0`` disables spreading.
        """

        if alpha <= 0.0:
            return
        new_rho: Dict[Tuple[str, str], float] = defaultdict(float, self._rho)
        for edge in graph.edges:
            key = (edge.source, edge.target)
            neighbours: set[Edge] = set()
            for nid in (edge.source, edge.target):
                neighbours.update(graph.get_edges_from(nid))
                neighbours.update(graph.get_edges_to(nid))
            if not neighbours:
                continue
            mean = sum(
                self._rho.get((e.source, e.target), 0.0) for e in neighbours
            ) / len(neighbours)
            new_rho[key] = (1 - alpha) * self._rho.get(key, 0.0) + alpha * mean
        self._rho = new_rho

    # ------------------------------------------------------------------
    def clear(self) -> None:
        """Reset all density values."""

        self._rho.clear()


# Singleton instance used across the engine
_density_field = DensityField()


def get_field() -> DensityField:
    """Return the global :class:`DensityField` instance."""

    return _density_field
