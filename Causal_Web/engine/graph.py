from .node import Node, Edge
import json

class CausalGraph:
    def __init__(self):
        self.nodes = {}
        self.edges = []

    def add_node(self, node_id, x=0.0, y=0.0, frequency=1.0):
        self.nodes[node_id] = Node(node_id, x, y, frequency)

    def add_edge(self, source_id, target_id, delay=1):
        self.edges.append(Edge(source_id, target_id, delay))

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

    def to_dict(self):
        return {
            "nodes": {
                nid: {
                    "x": n.x,
                    "y": n.y,
                    "ticks": [{"time": t, "phase": p} for t, p in n.tick_history]
                } for nid, n in self.nodes.items()
            },
            "edges": [
                {"from": e.source, "to": e.target, "delay": e.delay}
                for e in self.edges
            ]
        }

    def save_to_file(self, path):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


    def load_from_file(self, path):
        with open(path, 'r') as f:
            data = json.load(f)

        self.nodes.clear()
        self.edges.clear()

        for node_id, node_data in data.get("nodes", {}).items():
            self.add_node(
                node_id,
                x=node_data.get("x", 0.0),
                y=node_data.get("y", 0.0),
                frequency=node_data.get("frequency", 1.0)
            )

        for edge in data.get("edges", []):
            self.add_edge(edge["from"], edge["to"], delay=edge.get("delay", 1))
