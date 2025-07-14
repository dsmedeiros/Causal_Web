import cmath
import math

from engine.bridge import Bridge
from .node import Node, Edge
import json

class CausalGraph:
    def __init__(self):
        self.nodes = {}
        self.edges = []
        self.bridges = []
        self.tick_sources = []

    def add_node(self, node_id, x=0.0, y=0.0, frequency=1.0, refractory_period=2, base_threshold=0.5, phase=0.0):
        self.nodes[node_id] = Node(node_id, x, y, frequency, refractory_period, base_threshold, phase)

    def add_edge(self, source_id, target_id, attenuation=1.0, density=0.0, delay=1, phase_shift=0.0):
        self.edges.append(Edge(source_id, target_id, attenuation, density, delay, phase_shift))

    def add_bridge(self, node_a_id, node_b_id, bridge_type="braided", phase_offset=0.0,
               drift_tolerance=None, decoherence_limit=None):
        self.bridges.append(Bridge(node_a_id, node_b_id, bridge_type, phase_offset,
                               drift_tolerance, decoherence_limit))

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
                    complex_phase = [cmath.rect(1.0, p % (2 * math.pi)) for p in raw_phases]
                    vector_sum = sum(complex_phase)
                    contributors = [
                        {
                            "phase": round(p % (2 * math.pi), 4),
                            "magnitude": 1.0
                        }
                        for p in raw_phases
                    ]
                    result = {
                        "tick": int(tick),
                        "node": node.id,
                        "contributors": contributors,
                        "interference_result": {
                            "resultant_phase": round(cmath.phase(vector_sum) % (2 * math.pi), 4),
                            "amplitude": round(abs(vector_sum), 4),
                            "type": self._interference_type(raw_phases)
                        },
                        "collapsed": any(tick_obj.time == tick for tick_obj in node.tick_history),
                        "bridge_status": [
                            {
                                "between": [bridge.node_a_id, bridge.node_b_id],
                                "active": bridge.active,
                                "type": bridge.bridge_type,
                                "drift_tolerance": bridge.drift_tolerance,
                                "decoherence_limit": bridge.decoherence_limit
                            }
                            for bridge in self.bridges
                            if node.id in (bridge.node_a_id, bridge.node_b_id)
                        ]
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
            for p2 in normalized[i+1:]
        ]
        if all(d < 0.1 for d in phase_diffs):
            return "constructive"
        elif all(abs(d - math.pi) < 0.1 for d in phase_diffs):
            return "destructive"
        else:
            return "partial"

    def to_dict(self):
        return {
            "nodes": {
                nid: {
                    "x": n.x,
                    "y": n.y,
                    "ticks": [{"time": tick.time, "phase": tick.phase, "origin": tick.origin} for tick in n.tick_history],
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
                    "subjective_ticks": n.subjective_ticks
                } for nid, n in self.nodes.items()
            },
            "superpositions": {
                nid: {
                    str(t): [round(float(p), 4) for p in node.pending_superpositions[t]]
                    for t in node.pending_superpositions
                }
                for nid, node in self.nodes.items() if node.pending_superpositions
            },
            "edges": [
                {
                    "from": e.source,
                    "to": e.target,
                    "delay": e.delay,
                    "attenuation": e.attenuation,
                    "density": e.density,
                    "phase_shift": e.phase_shift
                }
                for e in self.edges
            ],
            "tick_sources": self.tick_sources
        }

    def save_to_file(self, path):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


    def load_from_file(self, path):
        with open(path, 'r') as f:
            data = json.load(f)

        self.nodes.clear()
        self.edges.clear()
        self.bridges.clear()

        for node_data in data.get("nodes", []):
            self.add_node(
                node_data["id"],
                x=node_data.get("x", 0.0),
                y=node_data.get("y", 0.0),
                frequency=node_data.get("frequency", 1.0),
                refractory_period=node_data.get("refractory_period", 2.0),
                base_threshold=node_data.get("base_threshold", 0.5),
                phase=node_data.get("phase", 0.0)
            )

        for edge in data.get("edges", []):
            self.add_edge(
                edge["from"], 
                edge["to"],
                attenuation=edge.get("attenuation", 1.0),
                density=edge.get("density", 0.0),
                delay=edge.get("delay", 1),
                phase_shift=edge.get("phase_shift", 0.0))

        for bridge in data.get("bridges", []):
            self.add_bridge(
                bridge["from"],
                bridge["to"],
                bridge_type=bridge.get("bridge_type", "braided"),
                phase_offset=bridge.get("phase_offset", 0.0),
                drift_tolerance=bridge.get("drift_tolerance"),
                decoherence_limit=bridge.get("decoherence_limit")
            )

        self.tick_sources = data.get("tick_sources", [])

