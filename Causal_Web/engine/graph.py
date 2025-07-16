import cmath
import math
from collections import defaultdict

from .bridge import Bridge
from .node import Node, Edge, NodeType
from .meta_node import MetaNode
from ..config import Config
import json
from .logger import log_json


class CausalGraph:
    """Container for nodes, edges and bridges comprising the simulation."""

    def __init__(self):
        self.nodes = {}
        self.edges = []
        self.bridges = []
        self.tick_sources = []
        self.meta_nodes = {}
        self.spatial_index = defaultdict(set)

    def add_node(
        self,
        node_id,
        x=0.0,
        y=0.0,
        frequency=1.0,
        refractory_period=2,
        base_threshold=0.5,
        phase=0.0,
        *,
        origin_type="seed",
        generation_tick=0,
        parent_ids=None,
    ):
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

    def add_edge(
        self,
        source_id,
        target_id,
        attenuation=1.0,
        density=0.0,
        delay=1,
        phase_shift=0.0,
    ):
        self.edges.append(
            Edge(source_id, target_id, attenuation, density, delay, phase_shift)
        )

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
        self.bridges.append(
            Bridge(
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
        )

    def get_node(self, node_id):
        return self.nodes.get(node_id)

    def get_edges_from(self, node_id):
        return [e for e in self.edges if e.source == node_id]

    def get_edges_to(self, node_id):
        return [e for e in self.edges if e.target == node_id]

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
        return [self.nodes[i] for i in ids]

    # --- Bridge-aware connectivity helpers ---
    def get_bridge_neighbors(self, node_id, active_only=True):
        """Return IDs of nodes connected via bridges."""
        neighbors = []
        for b in self.bridges:
            if active_only and not b.active:
                continue
            if b.node_a_id == node_id:
                neighbors.append(b.node_b_id)
            elif b.node_b_id == node_id:
                neighbors.append(b.node_a_id)
        return neighbors

    def get_connected_nodes(self, node_id):
        """All neighbors reachable via edges or active bridges."""
        connected = set(
            self.get_upstream_nodes(node_id) + self.get_downstream_nodes(node_id)
        )
        connected.update(self.get_bridge_neighbors(node_id))
        return list(connected)

    def reset_ticks(self):
        for node in self.nodes.values():
            node.tick_history.clear()
            node.incoming_phase_queue.clear()
            node.current_tick = 0
            node.subjective_ticks = 0
            node.last_emission_tick = None
            node.coherence = 1.0
            node.decoherence = 0.0

    def inspect_superpositions(self):
        inspection_log = []

        for node in self.nodes.values():
            for tick, raw_phases in node.pending_superpositions.items():
                if len(raw_phases) > 1:
                    # normalize each float to complex unit vector
                    complex_phase = [
                        cmath.rect(1.0, p % (2 * math.pi)) for p in raw_phases
                    ]
                    vector_sum = sum(complex_phase)
                    contributors = [
                        {"phase": round(p % (2 * math.pi), 4), "magnitude": 1.0}
                        for p in raw_phases
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
        """Detect sets of phase-aligned nodes."""
        clusters = []
        visited = set()
        node_list = list(self.nodes.values())
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
                ):
                    cluster.append(other.id)
                    visited.add(other.id)
            if len(cluster) > 1:
                clusters.append(cluster)
        return clusters

    def create_meta_nodes(self, clusters):
        """Instantiate MetaNode objects for given clusters."""
        for cluster in clusters:
            meta = MetaNode(cluster, self)
            self.meta_nodes[meta.id] = meta

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
                    str(t): [round(float(p), 4) for p in node.pending_superpositions[t]]
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
        }

    def save_to_file(self, path):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def load_from_file(self, path):
        with open(path, "r") as f:
            data = json.load(f)

        self.nodes.clear()
        self.edges.clear()
        self.bridges.clear()
        self.tick_sources = []

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
                refractory_period=node_data.get("refractory_period", 2.0),
                base_threshold=node_data.get("base_threshold", 0.5),
                phase=node_data.get("phase", 0.0),
                origin_type=node_data.get("origin_type", "seed"),
                generation_tick=node_data.get("generation_tick", 0),
                parent_ids=node_data.get("parent_ids"),
            )
            goals = node_data.get("goals")
            if goals is not None:
                self.nodes[node_id].goals = goals
        for edge in data.get("edges", []):
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
            with open(Config.output_path("void_node_map.json"), "w") as f:
                json.dump(self.void_nodes, f, indent=2)
        with open(Config.output_path("connectivity_log.json"), "w") as f:
            json.dump(connectivity_log, f, indent=2)

    # ------------------------------------------------------------
    def emit_law_wave(self, origin_id: str, tick: int, radius: int = 2) -> None:
        """Propagate a law wave from origin node and log affected nodes."""
        visited = {origin_id}
        frontier = [(origin_id, 0)]
        affected = []
        while frontier:
            nid, dist = frontier.pop(0)
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
