from __future__ import annotations

"""Delay mapping strategies."""

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable


def log_scalar(rho: float, *, d0: float, gamma: float, rho0: float) -> int:
    """Logarithmic scalar delay map.

    Parameters
    ----------
    rho:
        Local density value.
    d0, gamma, rho0:
        Baseline delay and scaling parameters.
    """

    return int(max(1, d0 + math.floor(gamma * math.log(1 + rho / rho0))))


@dataclass
class PhiLinear:
    """Φ-based linear delay map maintaining per-vertex scalars."""

    omega: float = 0.2
    alpha: float = 0.05
    eta: float = 0.1
    init: float = 0.0
    phi: Dict[int, float] = field(default_factory=dict)

    def update_vertex(self, v: int, neighbours: Iterable[int], rho_bar: float) -> float:
        """Update the Φ value for vertex ``v``.

        Parameters
        ----------
        v:
            Vertex identifier.
        neighbours:
            Iterable of neighbouring vertex identifiers.
        rho_bar:
            Mean incident density for ``v``.
        """

        neighbour_list = list(neighbours)
        deg = max(1, len(neighbour_list))
        avg_phi = (
            sum(self.phi.get(u, self.init) for u in neighbour_list) / deg
            if neighbour_list
            else 0.0
        )
        phi_v = (
            (1 - self.omega) * self.phi.get(v, self.init)
            + self.omega * avg_phi
            + self.eta * rho_bar
        )
        self.phi[v] = phi_v
        return phi_v

    def effective_delay(self, d0: float, u: int, v: int) -> int:
        """Compute ``d_eff`` for edge ``(u, v)`` using Φ values."""

        phi_u = self.phi.get(u, self.init)
        phi_v = self.phi.get(v, self.init)
        deff = d0 * (1 + self.alpha * (phi_u + phi_v) / 2.0)
        return int(max(1, math.floor(deff)))


def stamp_delay_metadata(
    meta: Dict[str, object], mode: str, params: Dict[str, object]
) -> None:
    """Annotate run metadata with the active delay map and parameters."""

    meta["delay_map"] = {"mode": mode, **params}


__all__ = ["log_scalar", "PhiLinear", "stamp_delay_metadata"]
