"""Dynamic ε-pair management for the v2 engine.

This module implements a toy model for *ε*-pair formation.  Each
"seed" carries a time-to-live (TTL), a hash prefix identifying the
originating site and a local ``theta`` value.  Seeds are emitted along
outgoing edges during Q-delivery and, when two compatible seeds meet,
their sources are connected by a temporary "bridge" edge.  Bridges
decay unless periodically reinforced.

The implementation is intentionally lightweight – it exists to provide a
concrete API for higher level components while the physical model is
still under development.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np

from ..logging.logger import log_record


@dataclass
class Seed:
    """A propagating ε-pair seed.

    Parameters
    ----------
    origin:
        Identifier of the node that emitted the seed.
    ttl:
        Remaining hop budget for the seed.
    h_prefix:
        ``L``-bit prefix derived from the origin's hash.
    theta:
        Local phase value associated with the seed.
    """

    origin: int
    ttl: int
    h_prefix: int
    theta: float


@dataclass
class Bridge:
    """State associated with a dynamic bridge.

    Attributes
    ----------
    sigma:
        Reinforcement level for the bridge.
    edge_id:
        Synthetic identifier used when scheduling packets across the
        bridge.  A unique negative ID is allocated per bridge and
        remains stable until the bridge is removed, ensuring the ID
        space does not clash with real edges.
    """

    sigma: float
    edge_id: int = -1


class EPairs:
    """Manage ε-pair seeds and bridges.

    The class exposes three core operations:

    ``emit``
        Emit seeds from a node to a set of neighbours.
    ``reinforce``
        Update the ``sigma`` value for an existing bridge when a tick is
        delivered across it, removing the bridge if it decays below
        ``sigma_min``.
    ``bridges``
        Mapping of ``(a, b)`` node id pairs to :class:`Bridge` objects.

    ``bridge_created`` and ``bridge_removed`` events are logged via the
    global logger when bridges form or decay below ``sigma_min``.

    Parameters are supplied on construction to avoid global state and to
    make the component easy to test. A ``seed`` may be provided for
    deterministic behaviour in routines that rely on randomness.
    """

    def __init__(
        self,
        delta_ttl: int,
        ancestry_prefix_L: int,
        theta_max: float,
        sigma0: float,
        lambda_decay: float,
        sigma_reinforce: float,
        sigma_min: float,
        seed: int | None = None,
    ) -> None:
        self.delta_ttl = delta_ttl
        self.L = ancestry_prefix_L
        self.theta_max = theta_max
        self.sigma0 = sigma0
        self.lambda_decay = lambda_decay
        self.sigma_reinforce = sigma_reinforce
        self.sigma_min = sigma_min
        self.seeds: Dict[int, List[Seed]] = {}
        self.bridges: Dict[Tuple[int, int], Bridge] = {}
        # adjacency list of active bridge partners
        self.adjacency: Dict[int, List[int]] = {}
        self._rng = np.random.default_rng(seed)
        # Synthetic edge identifier allocation for bridges
        self._next_bridge_id = -1

    # ------------------------------------------------------------------
    # seed handling

    def emit(
        self, origin: int, h_value: int, theta: float, neighbours: Iterable[int]
    ) -> None:
        """Emit seeds from ``origin`` to each neighbour.

        Each seed consumes one unit of TTL during the hop.  Seeds with a
        remaining TTL of zero are discarded.  When a seed arrives at a
        site it is compared against existing seeds to determine whether a
        bridge should be formed.
        """

        prefix = self._prefix(h_value)
        for n in neighbours:
            seed = Seed(
                origin=origin, ttl=self.delta_ttl - 1, h_prefix=prefix, theta=theta
            )
            if seed.ttl > 0:
                self._place_seed(n, seed)

    def carry(self, site: int, neighbours: Iterable[int]) -> None:
        """Propagate existing seeds at ``site`` to its neighbours.

        Each hop consumes one unit of TTL. Seeds whose TTL drops to zero
        or below are discarded. Seeds are removed from ``site`` once they
        have been carried.
        """

        seeds = self.seeds.pop(site, [])
        for seed in seeds:
            if seed.ttl <= 0:
                continue
            ttl = seed.ttl - 1
            if ttl <= 0:
                continue
            for n in neighbours:
                self._place_seed(
                    n,
                    Seed(
                        origin=seed.origin,
                        ttl=ttl,
                        h_prefix=seed.h_prefix,
                        theta=seed.theta,
                    ),
                )

    # Internal helpers -------------------------------------------------

    def _prefix(self, h_value: int) -> int:
        if self.L <= 0:
            return 0
        shift = max(0, h_value.bit_length() - self.L)
        return h_value >> shift

    def _place_seed(self, site: int, seed: Seed) -> None:
        seeds = self.seeds.setdefault(site, [])
        for other in list(seeds):
            if (
                seed.h_prefix == other.h_prefix
                and abs(seed.theta - other.theta) <= self.theta_max
                and other.ttl > 0
            ):
                self._create_bridge(seed.origin, other.origin)
                seeds.remove(other)
                return
        seeds.append(seed)

    def _create_bridge(self, a: int, b: int) -> None:
        key = self._bridge_key(a, b)
        if key not in self.bridges:
            edge_id = self._next_bridge_id
            self.bridges[key] = Bridge(self.sigma0, edge_id)
            self._next_bridge_id -= 1
            self.adjacency.setdefault(a, []).append(b)
            self.adjacency.setdefault(b, []).append(a)
            log_record(
                category="event",
                label="bridge_created",
                value={
                    "src": a,
                    "dst": b,
                    "sigma": self.sigma0,
                    "bridge_id": edge_id,
                },
            )

    def _bridge_key(self, a: int, b: int) -> Tuple[int, int]:
        return (a, b) if a <= b else (b, a)

    def _remove_bridge(self, a: int, b: int) -> None:
        key = self._bridge_key(a, b)
        bridge = self.bridges.get(key)
        if bridge is not None:
            log_record(
                category="event",
                label="bridge_removed",
                value={
                    "src": a,
                    "dst": b,
                    "sigma": bridge.sigma,
                    "bridge_id": bridge.edge_id,
                },
            )
            del self.bridges[key]
        for src, dst in ((a, b), (b, a)):
            neigh = self.adjacency.get(src)
            if neigh and dst in neigh:
                neigh.remove(dst)
                if not neigh:
                    del self.adjacency[src]

    # ------------------------------------------------------------------
    # bridge handling

    def reinforce(self, a: int, b: int) -> None:
        """Reinforce the bridge between ``a`` and ``b`` if present."""

        key = self._bridge_key(a, b)
        bridge = self.bridges.get(key)
        if bridge is None:
            return
        bridge.sigma += self.sigma_reinforce
        if bridge.sigma < self.sigma_min:
            self._remove_bridge(a, b)

    def decay_all(self) -> None:
        """Decay all bridges, removing those below :attr:`sigma_min`."""

        factor = 1.0 - self.lambda_decay
        if factor >= 1.0:
            return
        for (a, b), bridge in list(self.bridges.items()):
            bridge.sigma *= factor
            if bridge.sigma < self.sigma_min:
                self._remove_bridge(a, b)
