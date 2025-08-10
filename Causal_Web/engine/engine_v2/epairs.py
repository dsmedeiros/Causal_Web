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
    """State associated with a dynamic bridge."""

    sigma: float


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
        self._rng = np.random.default_rng(seed)

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
            self.bridges[key] = Bridge(self.sigma0)

    def _bridge_key(self, a: int, b: int) -> Tuple[int, int]:
        return (a, b) if a <= b else (b, a)

    # ------------------------------------------------------------------
    # bridge handling

    def reinforce(self, a: int, b: int) -> None:
        """Reinforce or decay the bridge between ``a`` and ``b``."""

        key = self._bridge_key(a, b)
        bridge = self.bridges.get(key)
        if bridge is None:
            return
        bridge.sigma = (1.0 - self.lambda_decay) * bridge.sigma + self.sigma_reinforce
        if bridge.sigma < self.sigma_min:
            del self.bridges[key]
