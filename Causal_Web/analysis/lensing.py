"""Gravitational lensing analysis utilities."""

from __future__ import annotations

import random
from typing import Hashable

import networkx as nx

from ..engine.analysis.mc_paths import monte_carlo_path_integral

__all__ = ["lensing_wedge_amplitude"]


def lensing_wedge_amplitude(
    graph: nx.DiGraph,
    source: Hashable,
    target: Hashable,
    *,
    k: int = 100,
    samples: int = 1000,
    weight: str = "delay",
    rng: random.Random | None = None,
) -> complex:
    """Estimate the lensing wedge amplitude between two nodes.

    This function is a thin wrapper over
    :func:`~Causal_Web.engine.analysis.mc_paths.monte_carlo_path_integral`
    used in performance-sensitive analyses. It enumerates the ``k`` shortest
    paths between ``source`` and ``target`` and returns a Monte-Carlo estimate
    of their aggregate complex amplitude.
    """

    return monte_carlo_path_integral(
        graph,
        source,
        target,
        k=k,
        samples=samples,
        weight=weight,
        rng=rng,
    )
