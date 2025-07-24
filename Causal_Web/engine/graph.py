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

    def __init__(self):
        self.nodes = {}
        self.edges = []
        self.edges_from = defaultdict(list)
        self.edges_to = defaultdict(list)
        self.bridges = []
        self.bridges_by_node = defaultdict(set)
        self.tick_sources = []
        self.meta_nodes: dict[str, MetaNode] = {}
        self.spatial_index = defaultdict(set)

    def add_node(
        self,
        node_id,
        x=0.0,
        y=0.0,
        frequency=1.0,
        refractory_period: float | None = None,
        base_threshold=0.5,
        phase=0.0,
        *,
        origin_type="seed",
        generation_tick=0,
        parent_ids=None,
    ):
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
        """Return coordinates shifted so no node is within ``gap`` units on both axes."""
        while any(
            abs(n.x - x) < gap and abs(n.y - y) < gap for n in self.nodes.values()
        ):
            x += gap
            y += gap
        return x, y

    def add_edge(
        self,
        source_id,
        target_id,
        attenuation=1.0,
        density=0.0,
        delay=1,
        phase_shift=0.0,
        weight=None,
    ):
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
        node_a_id,
        node_b_id,
        bridge_type="braided",
        phase_offset=0.0,
        drift_tolerance=None,
        decoherence_limit=None,
        initial_strength=1.0,
        medium_type="standard",
        mutable=True,
        seeded=True,
        formed_at_tick=0,
    ):
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

    def get_node(self, node_id):
        return self.nodes.get(node_id)

    def get_edges_from(self, node_id):
        return self.edges_from.get(node_id, [])

    def get_edges_to(self, node_id):
        return self.edges_to.get(node_id, [])

    def get_upstream_nodes(self, node_id):
        return [e.source for e in self.get_edges_to(node_id)]

    def get_downstream_nodes(self, node_id):
        return [e.target for e in self.get_edges_from(node_id)]

    def nearby_nodes(self, node, radius=1):
        """Return nodes in adjacent spatial partitions."""
        cells = [
            (node.grid_x + dx, node.grid_y + dy)
            for dx in range(-radius, radius + 1)
            for dy in range(-radius, radius + 1)
        ]
        ids = set()
        for c in cells:
            ids.update(self.spatial_index.get(c, set()))
        ids.discard(node.id)
        return [self.nodes[i] for i in ids if i in self.nodes]

    # --- Bridge-aware connectivity helpers ---
    def get_bridge_neighbors(self, node_id, active_only=True):
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

    def get_connected_nodes(self, node_id):
        """All neighbors reachable via edges or active bridges."""
        connected = set(
            self.get_upstream_nodes(node_id) + self.get_downstream_nodes(node_id)
        )
        connected.update(self.get_bridge_neighbors(node_id))
        return list(connected)

    def reset_ticks(self):
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

    def inspect_superpositions(self):
        inspection_log = []

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

    def _interference_type(self, phases):
        if len(phases) < 2:
            return "neutral"

        normalized = [p % (2 * math.pi) for p in phases]

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
    ):
        """Detect sets of phase-aligned nodes and assign cluster IDs."""
        clusters = []
        visited = set()
        node_list = list(self.nodes.values())
        cid = 0
        for node in node_list:
            if node.id in visited or node.law_wave_frequency == 0.0:
                continue
            cluster = [node.id]
            visited.add(node.id)
            for other in node_list:
                if other.id in visited or other.law_wave_frequency == 0.0:
                    continue
                if (
                    abs(node.law_wave_frequency - other.law_wave_frequency)
                    <= freq_tolerance
                    and node.coherence > coherence_threshold
                    and other.coherence > coherence_threshold
                    and abs(node.grid_x - other.grid_x) <= 1
                    and abs(node.grid_y - other.grid_y) <= 1
                ):
                    cluster.append(other.id)
                    visited.add(other.id)
            for nid in cluster:
                self.nodes[nid].cluster_ids[0] = cid
            cid += 1
            if len(cluster) > 1:
                clusters.append(cluster)
        for nid, node in self.nodes.items():
            if 0 not in node.cluster_ids:
                node.cluster_ids[0] = cid
                cid += 1
        return clusters

    def hierarchical_clusters(self) -> dict:
        """Compute hierarchical clustering assignments."""
        self.detect_clusters()
        visited = set()
        components = []
        cid = 0

        def neighbors(nid: str) -> list[str]:
            edge_n = [e.target for e in self.get_edges_from(nid)]
            edge_n += [e.source for e in self.get_edges_to(nid)]
            edge_n += self.get_bridge_neighbors(nid, active_only=True)
            return edge_n

        for nid in self.nodes:
            if nid in visited:
                continue
            queue = deque([nid])
            comp = []
            visited.add(nid)
            while queue:
                cur = queue.popleft()
                comp.append(cur)
                for nb in neighbors(cur):
                    if nb not in visited:
                        visited.add(nb)
                        queue.append(nb)
            for member in comp:
                self.nodes[member].cluster_ids[1] = cid
            components.append(comp)
            cid += 1
        return {0: [c for c in self._clusters_by_level(0)], 1: components}

    def _clusters_by_level(self, level: int) -> list:
        buckets: dict[int, list[str]] = {}
        for nid, node in self.nodes.items():
            cid = node.cluster_ids.get(level)
            if cid is None:
                continue
            buckets.setdefault(cid, []).append(nid)
        return list(buckets.values())

    def create_meta_nodes(self, clusters):
        """Instantiate MetaNode objects for given clusters."""
        for cluster in clusters:
            meta = MetaNode(cluster, self, meta_type="Emergent")
            self.meta_nodes[meta.id] = meta

    def update_meta_nodes(self, tick_time: int) -> None:
        """Update existing meta nodes and enforce their constraints."""
        for meta in list(self.meta_nodes.values()):
            meta.update_internal_state(tick_time)

    def to_dict(self):
        node_list = [
            {
                "id": nid,
                "x": n.x,
                "y": n.y,
                "ticks": [
                    {
                        "time": tick.time,
                        "phase": tick.phase,
                        "origin": tick.origin,
                        "layer": getattr(tick, "layer", "tick"),
                        "trace_id": getattr(tick, "trace_id", ""),
                    }
                    for tick in n.tick_history
                ],
                "phase": n.phase,
                "coherence": n.coherence,
                "decoherence": n.decoherence,
                "frequency": n.frequency,
                "refractory_period": n.refractory_period,
                "base_threshold": n.base_threshold,
                "collapse_origin": n.collapse_origin,
                "is_classical": getattr(n, "is_classical", False),
                "decoherence_streak": getattr(n, "_decoherence_streak", 0),
                "last_tick_time": n.last_tick_time,
                "subjective_ticks": n.subjective_ticks,
                "law_wave_frequency": n.law_wave_frequency,
                "trust_profile": n.trust_profile,
                "phase_confidence": n.phase_confidence_index,
                "goals": n.goals,
                "origin_type": n.origin_type,
                "generation_tick": n.generation_tick,
                "parent_ids": n.parent_ids,
                "node_type": n.node_type.value,
                "coherence_credit": n.coherence_credit,
                "decoherence_debt": n.decoherence_debt,
                "phase_lock": n.phase_lock,
            }
            for nid, n in self.nodes.items()
        ]

        return {
            "nodes": node_list,
            "superpositions": {
                nid: {
                    str(t): [
                        round(float(p[0] if isinstance(p, (tuple, list)) else p), 4)
                        for p in node.pending_superpositions[t]
                    ]
                    for t in node.pending_superpositions
                }
                for nid, node in self.nodes.items()
                if node.pending_superpositions
            },
            "edges": [
                {
                    "from": e.source,
                    "to": e.target,
                    "delay": e.delay,
                    "attenuation": e.attenuation,
                    "density": e.density,
                    "phase_shift": e.phase_shift,
                }
                for e in self.edges
            ],
            "bridges": [b.to_dict() for b in self.bridges],
            "tick_sources": self.tick_sources,
            "meta_nodes": {
                mid: {
                    "members": meta.member_ids,
                    "constraints": meta.constraints,
                    "type": meta.meta_type,
                    "origin": meta.origin,
                    "collapsed": meta.collapsed,
                    "x": meta.x,
                    "y": meta.y,
                }
                for mid, meta in self.meta_nodes.items()
            },
        }

    def save_to_file(self, path):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def load_from_file(self, path):
        """Populate the graph from a JSON file.

        The ``nodes`` section may be either a list of node objects or a
        dictionary mapping IDs to attribute dictionaries.  Likewise, ``edges``
        can be provided as a list of edge objects or as an adjacency mapping of
        ``{source: [target, ...]}`` or ``{source: {target: {params}}}``.
        """

        with open(path, "r") as f:
            data = json.load(f)

        self.nodes.clear()
        self.edges.clear()
        self.edges_from.clear()
        self.edges_to.clear()
        self.bridges.clear()
        self.bridges_by_node.clear()
        self.tick_sources = []
        self.spatial_index.clear()
        self.meta_nodes.clear()

        nodes_data = data.get("nodes", [])
        if isinstance(nodes_data, dict):
            nodes_iter = [dict(v, id=k) for k, v in nodes_data.items()]
        else:
            nodes_iter = nodes_data

        for node_data in nodes_iter:
            node_id = node_data.get("id")
            if node_id is None:
                continue
            self.add_node(
                node_id,
                x=node_data.get("x", 0.0),
                y=node_data.get("y", 0.0),
                frequency=node_data.get("frequency", 1.0),
                refractory_period=node_data.get(
                    "refractory_period", getattr(Config, "refractory_period", 2.0)
                ),
                base_threshold=node_data.get("base_threshold", 0.5),
                phase=node_data.get("phase", 0.0),
                origin_type=node_data.get("origin_type", "seed"),
                generation_tick=node_data.get("generation_tick", 0),
                parent_ids=node_data.get("parent_ids"),
            )
            goals = node_data.get("goals")
            if goals is not None:
                self.nodes[node_id].goals = goals
        edges_data = data.get("edges", [])
        if isinstance(edges_data, dict):
            edges_iter = []
            for src, targets in edges_data.items():
                if isinstance(targets, dict):
                    for tgt, params in targets.items():
                        rec = {"from": src, "to": tgt}
                        if isinstance(params, dict):
                            rec.update(params)
                        edges_iter.append(rec)
                else:
                    for tgt in targets:
                        if isinstance(tgt, str):
                            edges_iter.append({"from": src, "to": tgt})
                        elif isinstance(tgt, dict):
                            rec = {"from": src}
                            rec.update(tgt)
                            edges_iter.append(rec)
        else:
            edges_iter = edges_data

        for edge in edges_iter:
            src = edge.get("from")
            tgt = edge.get("to")
            if src is None or tgt is None:
                continue
            if src == tgt:
                # treat self-edge as tick seed definition
                self.tick_sources.append(
                    {
                        "node_id": src,
                        "tick_interval": edge.get("delay", 1),
                        "phase": edge.get("phase_shift", 0.0),
                    }
                )
                continue
            self.add_edge(
                src,
                tgt,
                attenuation=edge.get("attenuation", 1.0),
                density=edge.get("density", 0.0),
                delay=edge.get("delay", 1),
                phase_shift=edge.get("phase_shift", 0.0),
                weight=edge.get("weight"),
            )

        for bridge in data.get("bridges", []):
            src = bridge.get("from")
            tgt = bridge.get("to")
            if src is None or tgt is None:
                continue
            self.add_bridge(
                src,
                tgt,
                bridge_type=bridge.get("bridge_type", "braided"),
                phase_offset=bridge.get("phase_offset", 0.0),
                drift_tolerance=bridge.get("drift_tolerance"),
                decoherence_limit=bridge.get("decoherence_limit"),
                initial_strength=bridge.get("initial_strength", 1.0),
                medium_type=bridge.get("medium_type", "standard"),
                mutable=bridge.get("mutable", True),
                seeded=True,
                formed_at_tick=0,
            )

        self.tick_sources.extend(data.get("tick_sources", []))

        # Load configured meta nodes ---------------------------------
        for mid, meta in data.get("meta_nodes", {}).items():
            members = list(meta.get("members", []))
            constraints = dict(meta.get("constraints", {}))
            mtype = meta.get("type", "Configured")
            origin = meta.get("origin")
            collapsed = bool(meta.get("collapsed", False))
            x = float(meta.get("x", 0.0))
            y = float(meta.get("y", 0.0))
            self.meta_nodes[mid] = MetaNode(
                members,
                self,
                meta_type=mtype,
                constraints=constraints,
                origin=origin,
                collapsed=collapsed,
                x=x,
                y=y,
                id=mid,
            )

        self.identify_boundaries()

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
