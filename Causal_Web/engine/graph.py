from .node import Node, Edge

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