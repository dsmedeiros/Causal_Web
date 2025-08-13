from __future__ import annotations

import math
from typing import Any, Callable, Iterable, Mapping

from ..config import Config
from .fields.density import get_field
from .horizon import step as horizon_step
from .backend import ray_cluster
from .backend.zone_partitioner import partition_zones


def update_proper_time(node: Any, dt: float, rho: float, kappa: float) -> float:
    """Advance ``node.tau`` using velocity and local density.

    Parameters
    ----------
    node:
        Object with ``x``, ``y`` and ``tau`` attributes being updated.
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
    nodes: Iterable[Any],
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
    updating node clocks. Classical nodes are sharded into coherent zones and
    updated in parallel via :mod:`ray` when available.
    """

    nodes = list(nodes)
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

        zones = partition_zones(graph)

        def _update(zone: set[str]) -> None:
            for nid in zone:
                node = graph.get_node(nid)
                if node is None:
                    continue
                rho = rho_map.get(nid, 0.0)
                update_proper_time(node, dt, rho, kappa)

        ray_cluster.map_zones(_update, zones)
        processed = set().union(*zones)
    else:
        processed = set()

    for node in nodes:
        if node.id in processed:
            continue
        rho = rho_map.get(node.id, 0.0)
        update_proper_time(node, dt, rho, kappa)

    horizon_step(nodes)


def run_multi_layer(
    q_tick: Callable[[], None],
    c_tick: Callable[[], None],
    *,
    micro_ticks: int,
    macro_ticks: int,
    flush: Callable[[], None],
) -> None:
    """Execute quantum and classical layers in a nested schedule.

    Parameters
    ----------
    q_tick:
        Callback invoked for every quantum-layer micro-tick.
    c_tick:
        Callback invoked once per classical macro-tick after micro ticks.
    micro_ticks:
        Number of quantum micro iterations per classical macro iteration.
    macro_ticks:
        Total number of classical macro iterations to process.
    flush:
        Function called after ``micro_ticks`` to synchronize state between
        layers before the classical update.
    """

    for _ in range(macro_ticks):
        for _ in range(micro_ticks):
            q_tick()
        flush()
        c_tick()
