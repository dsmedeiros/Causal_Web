from __future__ import annotations

import cmath
import math
import random
from collections import defaultdict, deque

from .bridge import Bridge
from .node import Node, Edge, NodeType
from .tick import GLOBAL_TICK_POOL
from .meta_node import MetaNode
from ..config import Config
import json
from .logger import log_json


class CausalGraph:
    """Container for nodes, edges and bridges comprising the simulation."""

    def __init__(self) -> None:
        """Initialize empty collections tracking graph elements."""

        self.nodes = {}
        self.edges = []
        self.edges_from = defaultdict(list)
        self.edges_to = defaultdict(list)
        self.bridges = []
        self.bridges_by_node = defaultdict(set)
        self.tick_sources = []
        self.meta_nodes: dict[str, MetaNode] = {}
        self.spatial_index = defaultdict(set)
        self._cluster_cache: dict[int, tuple[int, list[list[str]]]] = {}
        self._cluster_version = 0
        self._nearby_cache: dict[tuple[str, int], list[str]] = {}
        self._nearby_cache_tick: int | None = None

    def add_node(
        self,
        node_id: str,
        x: float = 0.0,
        y: float = 0.0,
        frequency: float = 1.0,
        refractory_period: float | None = None,
        base_threshold: float = 0.5,
        phase: float = 0.0,
        *,
        origin_type: str = "seed",
        generation_tick: int = 0,
        parent_ids: list[str] | None = None,
    ) -> None:
        x, y = self._non_overlapping_position(x, y)
        if refractory_period is None:
            refractory_period = getattr(Config, "refractory_period", 2.0)
        self.nodes[node_id] = Node(
            node_id,
            x,
            y,
            frequency,
            refractory_period,
            base_threshold,
            phase,
            origin_type=origin_type,
            generation_tick=generation_tick,
            parent_ids=parent_ids,
        )
        node = self.nodes[node_id]
        self.spatial_index[(node.grid_x, node.grid_y)].add(node_id)

    def _non_overlapping_position(
        self, x: float, y: float, gap: int = 50
    ) -> tuple[float, float]:
        """Return coordinates shifted so no node lies within ``gap`` units."""

        cell_size = getattr(Config, "SPATIAL_GRID_SIZE", 50)
        radius = int(math.ceil(gap / cell_size))

        def _conflict(cx: int, cy: int) -> bool:
            cells = [
                (cx + dx, cy + dy)
                for dx in range(-radius, radius + 1)
                for dy in range(-radius, radius + 1)
            ]
            for c in cells:
                for nid in self.spatial_index.get(c, set()):
                    n = self.nodes.get(nid)
                    if n and abs(n.x - x) < gap and abs(n.y - y) < gap:
                        return True
            return False

        while True:
            cell_x = int(x // cell_size)
            cell_y = int(y // cell_size)
            if not _conflict(cell_x, cell_y):
                break
            x += gap
            y += gap
        return x, y

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        attenuation: float = 1.0,
        density: float = 0.0,
        delay: int = 1,
        phase_shift: float = 0.0,
        weight: float | None = None,
    ) -> None:
        if weight is None:
            low, high = getattr(Config, "edge_weight_range", [1.0, 1.0])
            weight = random.uniform(low, high)
        edge = Edge(
            source_id,
            target_id,
            attenuation,
            density,
            delay,
            phase_shift,
            weight,
        )
        self.edges.append(edge)
        self.edges_from[source_id].append(edge)
        self.edges_to[target_id].append(edge)

    def add_bridge(
        self,
        node_a_id: str,
        node_b_id: str,
        bridge_type: str = "braided",
        phase_offset: float = 0.0,
        drift_tolerance: float | None = None,
        decoherence_limit: float | None = None,
        initial_strength: float = 1.0,
        medium_type: str = "standard",
        mutable: bool = True,
        seeded: bool = True,
        formed_at_tick: int = 0,
    ) -> None:
        bridge = Bridge(
            node_a_id,
            node_b_id,
            bridge_type,
            phase_offset,
            drift_tolerance,
            decoherence_limit,
            initial_strength,
            medium_type,
            mutable,
            seeded,
            formed_at_tick,
        )
        self.bridges.append(bridge)
        self.bridges_by_node[node_a_id].add(bridge)
        self.bridges_by_node[node_b_id].add(bridge)

    def remove_node(self, node_id: str) -> None:
        """Remove a node and all references to it from the graph."""
        node = self.nodes.pop(node_id, None)
        if not node:
            return

        # Remove edges referencing this node
        self.edges = [
            e for e in self.edges if e.source != node_id and e.target != node_id
        ]
        self.edges_from.pop(node_id, None)
        self.edges_to.pop(node_id, None)

        for src, edges in list(self.edges_from.items()):
            self.edges_from[src] = [e for e in edges if e.target != node_id]
            if not self.edges_from[src]:
                self.edges_from.pop(src)

        for tgt, edges in list(self.edges_to.items()):
            self.edges_to[tgt] = [e for e in edges if e.source != node_id]
            if not self.edges_to[tgt]:
                self.edges_to.pop(tgt)

        # Remove bridges connected to this node
        for bridge in list(self.bridges):
            if node_id in (bridge.node_a_id, bridge.node_b_id):
                self.bridges.remove(bridge)
                self.bridges_by_node[bridge.node_a_id].discard(bridge)
                self.bridges_by_node[bridge.node_b_id].discard(bridge)
        self.bridges_by_node.pop(node_id, None)

        # Remove spatial index entry
        cell = (node.grid_x, node.grid_y)
        if cell in self.spatial_index:
            self.spatial_index[cell].discard(node_id)
            if not self.spatial_index[cell]:
                del self.spatial_index[cell]

    def get_node(self, node_id: str) -> Node | None:
        """Return the :class:`Node` with ``node_id`` if present."""
        return self.nodes.get(node_id)

    def get_edges_from(self, node_id: str) -> list[Edge]:
        """Return outbound edges from ``node_id``."""
        return self.edges_from.get(node_id, [])

    def set_current_tick(self, tick: int | None) -> None:
        """Reset neighborhood cache for a new tick."""
        if tick != self._nearby_cache_tick:
            self._nearby_cache_tick = tick
            self._nearby_cache.clear()

    def get_edges_to(self, node_id: str) -> list[Edge]:
        """Return inbound edges to ``node_id``."""
        return self.edges_to.get(node_id, [])

    def get_upstream_nodes(self, node_id: str) -> list[str]:
        """IDs of nodes with edges into ``node_id``."""
        return [e.source for e in self.get_edges_to(node_id)]

    def get_downstream_nodes(self, node_id: str) -> list[str]:
        """IDs of nodes reachable from ``node_id`` via edges."""
        return [e.target for e in self.get_edges_from(node_id)]

    def nearby_nodes(self, node: Node, radius: int = 1) -> list[Node]:
        """Return nodes in adjacent spatial partitions."""

        cache_key = (node.id, radius)
        if cache_key in self._nearby_cache:
            ids = self._nearby_cache[cache_key]
        else:
            cells = [
                (node.grid_x + dx, node.grid_y + dy)
                for dx in range(-radius, radius + 1)
                for dy in range(-radius, radius + 1)
            ]
            ids = set()
            for c in cells:
                ids.update(self.spatial_index.get(c, set()))
            ids.discard(node.id)
            ids = [i for i in ids if i in self.nodes]
            self._nearby_cache[cache_key] = ids
        return [self.nodes[i] for i in ids]

    # --- Bridge-aware connectivity helpers ---
    def get_bridge_neighbors(self, node_id: str, active_only: bool = True) -> list[str]:
        """Return IDs of nodes connected via bridges."""
        neighbors = set()
        for b in self.bridges_by_node.get(node_id, set()):
            if active_only and not b.active:
                continue
            if b.node_a_id == node_id:
                neighbors.add(b.node_b_id)
            else:
                neighbors.add(b.node_a_id)
        return list(neighbors)

    def get_connected_nodes(self, node_id: str) -> list[str]:
        """All neighbors reachable via edges or active bridges."""
        connected = set(
            self.get_upstream_nodes(node_id) + self.get_downstream_nodes(node_id)
        )
        connected.update(self.get_bridge_neighbors(node_id))
        return list(connected)

    def reset_ticks(self) -> None:
        for node in self.nodes.values():
            for t in node.tick_history:
                GLOBAL_TICK_POOL.release(t)
            node.tick_history.clear()
            node.emitted_tick_times.clear()
            node.received_tick_times.clear()
            node._tick_phase_lookup.clear()
            node._phase_cache.clear()
            node._coherence_cache.clear()
            node._decoherence_cache.clear()
            node.incoming_phase_queue.clear()
            node.pending_superpositions.clear()
            node.current_tick = 0
            node.subjective_ticks = 0
            node.last_emission_tick = None
            node.last_tick_time = None
            node.coherence = 1.0
            node.decoherence = 0.0

        # ensure no stale update flags remain
        try:
            from . import tick_engine as te

            te.nodes_to_update.clear()
        except Exception:
            pass

    def inspect_superpositions(self) -> list[dict]:
        """Return details of multi-phase superpositions on each node."""
        inspection_log: list[dict] = []

        for node in self.nodes.values():
            for tick, raw_phases in node.pending_superpositions.items():
                if len(raw_phases) > 1:
                    phases_only = [
                        p[0] if isinstance(p, (tuple, list)) else p for p in raw_phases
                    ]
                    # normalize each float to complex unit vector
                    complex_phase = [
                        cmath.rect(1.0, p % (2 * math.pi)) for p in phases_only
                    ]
                    vector_sum = sum(complex_phase)
                    contributors = [
                        {"phase": round(p % (2 * math.pi), 4), "magnitude": 1.0}
                        for p in phases_only
                    ]
                    result = {
                        "tick": int(tick),
                        "node": node.id,
                        "contributors": contributors,
                        "interference_result": {
                            "resultant_phase": round(
                                cmath.phase(vector_sum) % (2 * math.pi), 4
                            ),
                            "amplitude": round(abs(vector_sum), 4),
                            "type": self._interference_type(raw_phases),
                        },
                        "collapsed": any(
                            tick_obj.time == tick for tick_obj in node.tick_history
                        ),
                        "bridge_status": [
                            {
                                "between": [bridge.node_a_id, bridge.node_b_id],
                                "active": bridge.active,
                                "type": bridge.bridge_type,
                                "drift_tolerance": bridge.drift_tolerance,
                                "decoherence_limit": bridge.decoherence_limit,
                            }
                            for bridge in self.bridges
                            if node.id in (bridge.node_a_id, bridge.node_b_id)
                        ],
                    }
                    inspection_log.append(result)

        return inspection_log

    def _interference_type(self, phases: list) -> str:
        """Classify the interference pattern of a set of phases.

        Parameters
        ----------
        phases : list
            Sequence of phase values or ``(phase, tick)`` tuples.

        Returns
        -------
        str
            ``"constructive"`` when all phases are closely aligned,
            ``"destructive"`` when they cancel out, ``"partial``" otherwise.
        """

        phase_vals = [p[0] if isinstance(p, (tuple, list)) else p for p in phases]

        if len(phase_vals) < 2:
            return "neutral"

        normalized = [p % (2 * math.pi) for p in phase_vals]

        phase_diffs = [
            abs((p1 - p2 + math.pi) % (2 * math.pi) - math.pi)
            for i, p1 in enumerate(normalized)
            for p2 in normalized[i + 1 :]
        ]
        if all(d < 0.1 for d in phase_diffs):
            return "constructive"
        elif all(abs(d - math.pi) < 0.1 for d in phase_diffs):
            return "destructive"
        else:
            return "partial"

    def detect_clusters(
        self, coherence_threshold: float = 0.8, freq_tolerance: float = 0.1
    ) -> list[list[str]]:
        """Detect sets of phase-aligned nodes and assign cluster IDs.

        Nodes are compared only with others in adjacent spatial bins to
        reduce the number of pairwise checks.
        """

        # Ensure all nodes are present in the spatial index in case callers
        # bypassed :meth:`add_node` and inserted directly into ``nodes``.
        for nid, n in self.nodes.items():
            cell = (n.grid_x, n.grid_y)
            if nid not in self.spatial_index.get(cell, set()):
                self.spatial_index[cell].add(nid)

        clusters: list[list[str]] = []
        visited: set[str] = set()
        cid = 0

        for node in list(self.nodes.values()):
            if node.id in visited or node.law_wave_frequency == 0.0:
                continue

            cluster = [node.id]
            visited.add(node.id)

            if node.coherence > coherence_threshold:
                for other in self.nearby_nodes(node):
                    if (
                        other.id not in visited
                        and other.law_wave_frequency != 0.0
                        and abs(node.law_wave_frequency - other.law_wave_frequency)
                        <= freq_tolerance
                        and other.coherence > coherence_threshold
                    ):
                        cluster.append(other.id)
                        visited.add(other.id)

            for nid in cluster:
                if self.nodes[nid].cluster_ids.get(0) != cid:
                    self.nodes[nid].cluster_ids[0] = cid
            cid += 1
            if len(cluster) > 1:
                clusters.append(cluster)

        for nid, node in self.nodes.items():
            if 0 not in node.cluster_ids:
                self.nodes[nid].cluster_ids[0] = cid
                cid += 1

        self._cluster_version += 1
        self._cluster_cache.clear()

        return clusters

    def hierarchical_clusters(self) -> dict[int, list[list[str]]]:
        """Compute hierarchical clustering assignments.

        The first-level clusters must already be detected or they will be
        computed on demand. Neighborhood edges are cached for the duration of
        this call to avoid redundant lookups while traversing components.
        """
        if self._cluster_version == 0:
            self.detect_clusters()

        visited = set()
        components = []
        cid = 0

        neighbor_map: dict[str, list[str]] = {}
        for nid in self.nodes:
            edge_n = [e.target for e in self.get_edges_from(nid)]
            edge_n += [e.source for e in self.get_edges_to(nid)]
            edge_n += self.get_bridge_neighbors(nid, active_only=True)
            neighbor_map[nid] = edge_n

        for nid in self.nodes:
            if nid in visited:
                continue
            queue = deque([nid])
            comp = []
            visited.add(nid)
            while queue:
                cur = queue.popleft()
                comp.append(cur)
                for nb in neighbor_map.get(cur, []):
                    if nb not in visited:
                        visited.add(nb)
                        queue.append(nb)
            for member in comp:
                if self.nodes[member].cluster_ids.get(1) != cid:
                    self.nodes[member].cluster_ids[1] = cid
            components.append(comp)
            cid += 1

        self._cluster_version += 1
        self._cluster_cache.clear()
        return {0: [c for c in self._clusters_by_level(0)], 1: components}

    def _clusters_by_level(self, level: int) -> list[list[str]]:
        cached = self._cluster_cache.get(level)
        if cached and cached[0] == self._cluster_version:
            return cached[1]

        buckets: dict[int, list[str]] = {}
        for nid, node in self.nodes.items():
            cid = node.cluster_ids.get(level)
            if cid is None:
                continue
            buckets.setdefault(cid, []).append(nid)

        result = list(buckets.values())
        self._cluster_cache[level] = (self._cluster_version, result)
        return result

    def create_meta_nodes(self, clusters: list[list[str]]) -> None:
        """Instantiate MetaNode objects for given clusters."""
        for cluster in clusters:
            meta = MetaNode(cluster, self, meta_type="Emergent")
            self.meta_nodes[meta.id] = meta

    def update_meta_nodes(self, tick_time: int) -> None:
        """Update existing meta nodes and enforce their constraints."""
        for meta in list(self.meta_nodes.values()):
            meta.update_internal_state(tick_time)

    def to_dict(self) -> dict:
        """Return a JSON serializable representation of the graph."""

        from .serialization_service import GraphSerializationService

        return GraphSerializationService(self).as_dict()

    def save_to_file(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def load_from_file(self, path: str) -> None:
        """Populate the graph from a JSON file."""

        from .services.sim_services import GraphLoadService

        GraphLoadService(self, path).load()

    # ------------------------------------------------------------
    def identify_boundaries(self) -> None:
        self.void_nodes = []
        self.boundary_nodes = []
        connectivity_log = {}
        for nid, node in self.nodes.items():
            outgoing = self.get_edges_from(nid)
            incoming = self.get_edges_to(nid)
            bridges = self.get_bridge_neighbors(nid)
            total = len(outgoing) + len(incoming) + len(bridges)
            connectivity_log[nid] = {
                "edges_out": len(outgoing),
                "edges_in": len(incoming),
                "bridges": len(bridges),
                "total": total,
            }
            if total == 0:
                node.node_type = NodeType.NULL
                self.void_nodes.append(nid)
            if total <= 1:
                setattr(node, "boundary", True)
                self.boundary_nodes.append(nid)
            for b in bridges:
                other = self.get_node(b)
                if other and other.node_type == NodeType.NULL:
                    print(f"[WARNING] Bridge {nid}<->{b} connected to NULL node")
        if self.void_nodes:
            log_json(Config.output_path("void_node_map.json"), self.void_nodes)
        if Config.is_log_enabled("connectivity_log.json"):
            log_json(Config.output_path("connectivity_log.json"), connectivity_log)

    # ------------------------------------------------------------
    def emit_law_wave(self, origin_id: str, tick: int, radius: int = 2) -> None:
        """Propagate a law wave from origin node and log affected nodes."""
        visited = {origin_id}
        frontier = deque([(origin_id, 0)])
        affected = []
        while frontier:
            nid, dist = frontier.popleft()
            if dist > radius:
                continue
            node = self.get_node(nid)
            if node:
                node.collapse_pressure += max(0.0, 1.0 - 0.5 * dist)
                if nid != origin_id:
                    affected.append(nid)
            for edge in self.get_edges_from(nid):
                if edge.target not in visited:
                    visited.add(edge.target)
                    frontier.append((edge.target, dist + 1))
        if affected:
            log_json(
                Config.output_path("law_wave_log.json"),
                {"tick": tick, "origin": origin_id, "affected": affected},
            )
