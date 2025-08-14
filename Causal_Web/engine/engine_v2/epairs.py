"""Dynamic ε-pair management for the v2 engine.

This module implements a toy model for *ε*-pair formation.  Each
"seed" carries an expiry depth, a hash prefix identifying the
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
from typing import Callable, Dict, Iterable, List, Sequence, Set, Tuple
from array import array

import numpy as np
import logging

from ..logging.logger import log_record


@dataclass
class Seed:
    """A propagating ε-pair seed.

    Parameters
    ----------
    origin:
        Identifier of the node that emitted the seed.
    expiry_depth:
        Maximum propagation depth for the seed.
    h_prefix:
        ``L``-bit prefix derived from the origin's hash.
    theta:
        Local phase value associated with the seed.
    """

    origin: int
    expiry_depth: int
    h_prefix: int
    theta: float


@dataclass
class Bridge:
    """State associated with a dynamic bridge.

    Attributes
    ----------
    sigma:
        Reinforcement level for the bridge.
    d_bridge:
        Traversal delay applied when packets cross the bridge. The delay is
        derived from the incident edge delays of the bridge's endpoints and is
        set to ``max(1, median(d_eff))`` at creation time.
    edge_id:
        Synthetic identifier used when scheduling packets across the
        bridge.  A unique negative ID is allocated per bridge and
        remains stable until the bridge is removed, ensuring the ID
        space does not clash with real edges.
    """

    sigma: float
    d_bridge: int = 1
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

    ``overflow_drops``
        Counter of seeds dropped when a site's capacity limit is exceeded.

    ``bridge_created`` and ``bridge_removed`` events are logged via the
    global logger when bridges form or decay below ``sigma_min``.

    Parameters are supplied on construction to avoid global state and to
    make the component easy to test. A ``seed`` may be provided for
    deterministic behaviour in routines that rely on randomness. When
    omitted, the RNG is initialised nondeterministically.
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
        max_seeds_per_site: int = 64,
        sample_seed_rate: float = 0.01,
        sample_bridge_rate: float = 0.01,
    ) -> None:
        """Initialize the manager.

        Parameters
        ----------
        max_seeds_per_site:
            Maximum seeds to retain per site before oldest are evicted.
        sample_seed_rate:
            Probability that seed events are logged.
        sample_bridge_rate:
            Probability that bridge events are logged.
        """

        self.delta_ttl = delta_ttl
        self.L = ancestry_prefix_L
        self.theta_max = theta_max
        self.sigma0 = sigma0
        self.lambda_decay = lambda_decay
        self.sigma_reinforce = sigma_reinforce
        self.sigma_min = sigma_min
        self.max_seeds_per_site = max_seeds_per_site
        self.sample_seed_rate = sample_seed_rate
        self.sample_bridge_rate = sample_bridge_rate
        self.seeds: Dict[int, List[Seed]] = {}
        self.bridges: Dict[Tuple[int, int], Bridge] = {}
        # adjacency of active bridge partners stored in fixed arrays
        self.adjacency: Dict[int, array] = {}
        self._adj_free: Dict[int, List[int]] = {}
        self._rng = np.random.default_rng(seed)
        # Synthetic edge identifier allocation for bridges
        self._next_bridge_id = -1
        # Source for incident edge delays used when estimating bridge latency.
        self._incident_delays: Dict[int, List[int]] = {}
        self._incident_delay_cb: Callable[[int], Iterable[int]] | None = None
        # Track sites that have already emitted a capacity warning
        self._overflow_warned: Set[int] = set()
        self.overflow_drops: int = 0

    # Internal helpers -------------------------------------------------
    def _log_seed(self, label: str, value: Dict[str, int | float]) -> None:
        log_record(category="event", label=label, value=value)

    def _log_bridge(self, label: str, value: Dict[str, int | float]) -> None:
        log_record(category="event", label=label, value=value)

    def _add_adj(self, src: int, dst: int) -> None:
        arr = self.adjacency.setdefault(src, array("i"))
        free = self._adj_free.setdefault(src, [])
        if free:
            idx = free.pop()
            arr[idx] = dst
        else:
            arr.append(dst)

    def _del_adj(self, src: int, dst: int) -> None:
        arr = self.adjacency.get(src)
        if arr is None:
            return
        for i, val in enumerate(arr):
            if val == dst:
                arr[i] = -1
                self._adj_free.setdefault(src, []).append(i)
                break
        if all(v == -1 for v in arr):
            del self.adjacency[src]
            self._adj_free.pop(src, None)

    def partners(self, site: int) -> List[int]:
        """Return active bridge partners for ``site``."""

        arr = self.adjacency.get(site)
        if arr is None:
            return []
        return [v for v in arr if v != -1]

    # ------------------------------------------------------------------
    # seed handling

    def emit(
        self,
        origin: int,
        h_value: int,
        theta: float,
        depth_emit: int,
        edge_ids: Iterable[int],
        edges: Dict[str, Sequence[int]],
    ) -> None:
        """Emit seeds from ``origin`` to each neighbour.

        ``depth_emit`` is the depth of the emitting node.  Seeds inherit an
        ``expiry_depth`` of ``depth_emit + delta_ttl``.  A seed is only
        forwarded along an edge if the next hop depth ``depth_emit +
        edges['d_eff'][edge_id]`` does not exceed its expiry, providing a
        depth-based TTL that respects per-edge effective distance.
        """

        log_seeds = self._rng.random() < self.sample_seed_rate
        prefix = self._prefix(h_value)
        expiry = depth_emit + self.delta_ttl
        for edge_id in edge_ids:
            d_eff = int(edges["d_eff"][edge_id]) if "d_eff" in edges else 1
            depth_next = depth_emit + d_eff
            dst = int(edges["dst"][edge_id])
            if depth_next > expiry:
                if log_seeds:
                    self._log_seed(
                        "seed_dropped",
                        {"src": origin, "origin": origin, "reason": "expired"},
                    )
                continue
            if log_seeds:
                self._log_seed(
                    "seed_emitted",
                    {
                        "src": origin,
                        "dst": dst,
                        "origin": origin,
                        "expiry_depth": expiry,
                        "h_prefix": prefix,
                        "theta": theta,
                    },
                )
            self._place_seed(
                dst,
                Seed(
                    origin=origin,
                    expiry_depth=expiry,
                    h_prefix=prefix,
                    theta=theta,
                ),
                log_seeds,
            )

    def carry(
        self,
        site: int,
        depth_curr: int,
        edge_ids: Iterable[int],
        edges: Dict[str, Sequence[int]],
    ) -> None:
        """Propagate existing seeds at ``site`` to its neighbours.

        Seeds are removed from ``site`` once carried.  Forwarding along a
        given edge halts when the proposed ``depth_next`` would exceed a
        seed's ``expiry_depth``.
        """

        log_seeds = self._rng.random() < self.sample_seed_rate
        seeds = self.seeds.pop(site, [])
        for seed in seeds:
            for edge_id in edge_ids:
                d_eff = int(edges["d_eff"][edge_id]) if "d_eff" in edges else 1
                depth_next = depth_curr + d_eff
                if depth_next > seed.expiry_depth:
                    if log_seeds:
                        self._log_seed(
                            "seed_dropped",
                            {"src": site, "origin": seed.origin, "reason": "expired"},
                        )
                    continue
                dst = int(edges["dst"][edge_id])
                if log_seeds:
                    self._log_seed(
                        "seed_emitted",
                        {
                            "src": site,
                            "dst": dst,
                            "origin": seed.origin,
                            "expiry_depth": seed.expiry_depth,
                            "h_prefix": seed.h_prefix,
                            "theta": seed.theta,
                        },
                    )
                self._place_seed(
                    dst,
                    Seed(
                        origin=seed.origin,
                        expiry_depth=seed.expiry_depth,
                        h_prefix=seed.h_prefix,
                        theta=seed.theta,
                    ),
                    log_seeds,
                )

    # Internal helpers -------------------------------------------------

    def _prefix(self, h_value: int) -> int:
        """Return the first ``L`` MSBs of a 64-bit ancestry lane.

        Parameters
        ----------
        h_value:
            Value of the ``h0`` ancestry lane.  ``h_value`` is cast to
            ``uint64`` to ensure consistent semantics irrespective of the
            magnitude of the input integer.

        Returns
        -------
        int
            The ``L``-bit prefix extracted from the most significant bits of
            ``h_value``.  ``L`` values ``<= 0`` yield ``0``.
        """

        if self.L <= 0:
            return 0

        h0 = np.uint64(h_value)
        prefix = (h0 >> np.uint64(64 - self.L)) & np.uint64((1 << self.L) - 1)
        return int(prefix)

    def _place_seed(self, site: int, seed: Seed, log_seeds: bool) -> None:
        """Insert ``seed`` at ``site`` respecting capacity limits.

        Parameters
        ----------
        log_seeds:
            When ``True`` emit logging records for drop events.
        """

        seeds = self.seeds.setdefault(site, [])
        for other in list(seeds):
            if seed.h_prefix == other.h_prefix:
                if abs(seed.theta - other.theta) <= self.theta_max:
                    self._create_bridge(seed.origin, other.origin)
                    seeds.remove(other)
                else:
                    if log_seeds:
                        self._log_seed(
                            "seed_dropped",
                            {"src": site, "origin": seed.origin, "reason": "angle"},
                        )
                return
        if len(seeds) >= self.max_seeds_per_site:
            evicted = seeds.pop(0)
            self.overflow_drops += 1
            if log_seeds:
                self._log_seed(
                    "seed_dropped",
                    {"src": site, "origin": evicted.origin, "reason": "overflow"},
                )
            if site not in self._overflow_warned:
                logging.warning(
                    "max seeds per site reached at %s; dropping oldest", site
                )
                self._overflow_warned.add(site)
        seeds.append(seed)

    def set_incident_delays(
        self, delays: Dict[int, Iterable[int]] | Callable[[int], Iterable[int]]
    ) -> None:
        """Provide incident edge delays for bridge delay estimation.

        The source can be given either as a mapping from vertex id to a list of
        delays or as a callable returning an iterable of delays for a vertex.
        Supplying a callable allows the manager to read live delay values as the
        simulation evolves.
        """

        if callable(delays):
            self._incident_delay_cb = delays
            self._incident_delays = {}
        else:
            self._incident_delays = {v: list(vals) for v, vals in delays.items()}
            self._incident_delay_cb = None

    def _create_bridge(self, a: int, b: int, d_bridge: int | None = None) -> None:
        key = self._bridge_key(a, b)
        if key not in self.bridges:
            if d_bridge is None:
                if self._incident_delay_cb is not None:
                    delays_a = list(self._incident_delay_cb(a))
                    delays_b = list(self._incident_delay_cb(b))
                else:
                    delays_a = self._incident_delays.get(a, [])
                    delays_b = self._incident_delays.get(b, [])
                delays = delays_a + delays_b
                if delays:
                    d_bridge = max(1, int(np.median(delays)))
                else:
                    d_bridge = 1
            edge_id = self._next_bridge_id
            self.bridges[key] = Bridge(self.sigma0, d_bridge, edge_id)
            self._next_bridge_id -= 1
            self._add_adj(a, b)
            self._add_adj(b, a)
            if self._rng.random() < self.sample_bridge_rate:
                self._log_bridge(
                    "bridge_created",
                    {
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
            if self._rng.random() < self.sample_bridge_rate:
                self._log_bridge(
                    "bridge_removed",
                    {
                        "src": a,
                        "dst": b,
                        "sigma": bridge.sigma,
                        "bridge_id": bridge.edge_id,
                    },
                )
            del self.bridges[key]
        for src, dst in ((a, b), (b, a)):
            self._del_adj(src, dst)

    # ------------------------------------------------------------------
    # bridge handling

    def reinforce(self, a: int, b: int) -> None:
        """Reinforce the bridge between ``a`` and ``b`` if present.

        On each reinforcement the bridge delay is slowly recalibrated
        towards the current median of the incident edge delays for the two
        endpoints.  This allows long-lived bridges to track background
        drift in edge latency without rebuilding the bridge.
        """

        key = self._bridge_key(a, b)
        bridge = self.bridges.get(key)
        if bridge is None:
            return
        bridge.sigma += self.sigma_reinforce
        if bridge.sigma < self.sigma_min:
            self._remove_bridge(a, b)
            return

        if self._incident_delay_cb is not None:
            delays_a = list(self._incident_delay_cb(a))
            delays_b = list(self._incident_delay_cb(b))
        else:
            delays_a = self._incident_delays.get(a, [])
            delays_b = self._incident_delays.get(b, [])
        delays = delays_a + delays_b
        if delays:
            target = max(1, int(np.median(delays)))
            if target > bridge.d_bridge:
                bridge.d_bridge += 1
            elif target < bridge.d_bridge:
                bridge.d_bridge -= 1

    def adjust_d_bridge(self, site: int) -> None:
        """Nudge bridge delays incident on ``site`` toward local medians.

        Parameters
        ----------
        site:
            Vertex identifier whose incident edge delays were updated.
        """

        for partner in self.partners(site):
            key = self._bridge_key(site, partner)
            bridge = self.bridges.get(key)
            if bridge is None:
                continue
            if self._incident_delay_cb is not None:
                delays_site = list(self._incident_delay_cb(site))
                delays_partner = list(self._incident_delay_cb(partner))
            else:
                delays_site = self._incident_delays.get(site, [])
                delays_partner = self._incident_delays.get(partner, [])
            delays = delays_site + delays_partner
            if not delays:
                continue
            target = max(1, int(np.median(delays)))
            if target > bridge.d_bridge:
                bridge.d_bridge += 1
            elif target < bridge.d_bridge:
                bridge.d_bridge -= 1

    def decay_all(self) -> None:
        """Decay all bridges, removing those below :attr:`sigma_min`."""

        factor = 1.0 - self.lambda_decay
        if factor >= 1.0:
            return
        for (a, b), bridge in list(self.bridges.items()):
            bridge.sigma *= factor
            if bridge.sigma < self.sigma_min:
                self._remove_bridge(a, b)
