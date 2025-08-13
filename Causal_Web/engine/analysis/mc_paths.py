"""Monte-Carlo estimation of path integrals on causal graphs."""

from __future__ import annotations

import cmath
import random
from dataclasses import dataclass
from typing import Hashable, Iterable, List, Sequence

import networkx as nx

__all__ = [
    "PathInfo",
    "yen_k_shortest_paths",
    "accumulate_path",
    "monte_carlo_path_integral",
    "enumerate_path_integral",
]


@dataclass(frozen=True)
class PathInfo:
    """Container for path properties.

    Attributes
    ----------
    nodes:
        Sequence of node identifiers along the path.
    delay:
        Sum of edge ``delay`` attributes.
    phase:
        Sum of edge ``phase`` attributes.
    attenuation:
        Product of edge ``atten`` attributes.
    """

    nodes: Sequence[Hashable]
    delay: float
    phase: float
    attenuation: float

    @property
    def amplitude(self) -> complex:
        """Complex amplitude contributed by this path."""

        return self.attenuation * cmath.exp(1j * self.phase)


def yen_k_shortest_paths(
    graph: nx.DiGraph,
    source: Hashable,
    target: Hashable,
    k: int,
    weight: str = "delay",
) -> List[Sequence[Hashable]]:
    """Return up to ``k`` shortest simple paths from ``source`` to ``target``.

    The implementation delegates to :func:`networkx.shortest_simple_paths`,
    which internally uses Yen's algorithm to generate simple paths in
    nondecreasing order of total ``weight``.
    """

    paths: List[Sequence[Hashable]] = []
    generator = nx.shortest_simple_paths(graph, source, target, weight=weight)
    for _ in range(k):
        try:
            paths.append(next(generator))
        except StopIteration:
            break
    return paths


def accumulate_path(graph: nx.DiGraph, path: Sequence[Hashable]) -> PathInfo:
    """Accumulate delay, phase and attenuation for ``path``.

    Edge attributes may use either the short field names ``phase``/``atten`` or
    the legacy names ``phase_shift``/``attenuation``. Missing attributes default
    to ``0`` phase and ``1`` attenuation.

    Parameters
    ----------
    graph:
        Graph containing edges of ``path``.
    path:
        Sequence of node identifiers representing a simple path.
    """

    # TODO: legacy refactor

    delay = 0.0
    phase = 0.0
    attenuation = 1.0

    for u, v in nx.utils.pairwise(path):
        data = graph[u][v]
        delay += float(data.get("delay", 0.0))
        phase += float(data.get("phase", data.get("phase_shift", 0.0)))
        attenuation *= float(data.get("atten", data.get("attenuation", 1.0)))

    return PathInfo(list(path), delay, phase, attenuation)


def monte_carlo_path_integral(
    graph: nx.DiGraph,
    source: Hashable,
    target: Hashable,
    *,
    k: int = 100,
    samples: int = 1000,
    weight: str = "delay",
    rng: random.Random | None = None,
) -> complex:
    """Estimate the sum of complex amplitudes over paths from source to target.

    Paths are first truncated to the ``k`` shortest simple paths according to
    ``weight``. The returned value approximates the sum of amplitudes for this
    truncated set using Monte-Carlo sampling with ``samples`` draws.

    Edge attributes ``phase`` and ``atten`` control the complex contribution of
    each edge. Missing attributes default to ``0`` phase and ``1`` attenuation.

    Parameters
    ----------
    graph:
        Directed graph to sample.
    source, target:
        Nodes between which paths are considered.
    k:
        Maximum number of shortest paths to enumerate.
    samples:
        Number of Monte-Carlo samples to draw.
    weight:
        Edge attribute used as length for Yen's algorithm.
    rng:
        Optional :class:`random.Random` instance for reproducibility.

    Returns
    -------
    complex
        Estimated complex amplitude of the truncated path set.
    """

    rng = rng or random.Random()
    raw_paths = yen_k_shortest_paths(graph, source, target, k, weight=weight)
    infos = [accumulate_path(graph, p) for p in raw_paths]
    if not infos:
        return 0j

    amplitudes = [info.amplitude for info in infos]
    picks = rng.choices(amplitudes, k=samples)
    return sum(picks) / samples * len(amplitudes)


def enumerate_path_integral(
    graph: nx.DiGraph, source: Hashable, target: Hashable
) -> complex:
    """Compute the exact path integral by enumerating all simple paths.

    This helper is intended for small graphs and testing. The calculation uses
    :func:`networkx.all_simple_paths` and therefore may be extremely expensive
    for dense graphs.
    """

    total = 0j
    for path in nx.all_simple_paths(graph, source, target):
        info = accumulate_path(graph, path)
        total += info.amplitude
    return total
