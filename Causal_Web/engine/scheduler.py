from __future__ import annotations

import math
from typing import Iterable, Mapping

from ..config import Config
from .models.node import Node
from .fields.density import get_field
from .horizon import step as horizon_step


def update_proper_time(node: Node, dt: float, rho: float, kappa: float) -> float:
    """Advance ``node.tau`` using velocity and local density.

    Parameters
    ----------
    node:
        Node being updated.
    dt:
        Coordinate time step.
    rho:
        Local density associated with ``node``.
    kappa:
        Coupling constant that scales density impact.

    Returns
    -------
    float
        The proper-time increment ``d_tau`` applied to ``node``.
    """

    dx = node.x - getattr(node, "prev_x", node.x)
    dy = node.y - getattr(node, "prev_y", node.y)
    v2 = 0.0
    if dt > 0.0:
        v2 = (dx * dx + dy * dy) / (dt * dt)
    d_tau = dt * (1 - kappa * rho) * math.sqrt(max(0.0, 1 - v2))
    node.tau += d_tau
    node.prev_x = node.x
    node.prev_y = node.y
    return d_tau


def step(
    nodes: Iterable[Node],
    dt: float,
    rho_map: Mapping[str, float] | None = None,
    *,
    kappa: float | None = None,
    alpha: float | None = None,
    graph: "CausalGraph | None" = None,
) -> None:
    """Advance proper-time for ``nodes`` by ``dt``.

    ``rho_map`` may supply pre-computed densities keyed by node identifier. If
    omitted, zero density is assumed for every node. When ``graph`` is
    provided, the global density field is diffused using ``alpha`` weight before
    updating node clocks.
    """

    if rho_map is None:
        rho_map = {}
    if kappa is None:
        kappa = getattr(Config, "kappa", 0.0)
    if graph is not None:
        field = get_field()
        weight = (
            alpha
            if alpha is not None
            else getattr(Config, "density_diffusion_weight", 0.0)
        )
        field.diffuse(graph, weight)
    for node in nodes:
        rho = rho_map.get(node.id, 0.0)
        update_proper_time(node, dt, rho, kappa)
    horizon_step(nodes)
