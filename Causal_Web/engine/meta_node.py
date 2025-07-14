from typing import List


class MetaNode:
    """Abstract node representing a collapsed subgraph."""

    def __init__(self, node_ids: List[str], graph):
        self.id = "meta_" + "_".join(sorted(node_ids))
        self.member_ids = node_ids
        self.graph = graph
        self.phase = 0.0

    def apply_tick(self, tick_time: int, phase: float, origin: str = "meta"):
        """Propagate tick to all member nodes."""
        self.phase = phase
        for nid in self.member_ids:
            node = self.graph.get_node(nid)
            if node:
                node.apply_tick(tick_time, phase, self.graph, origin=origin)

    def update_internal_state(self, tick_time: int):
        for nid in self.member_ids:
            node = self.graph.get_node(nid)
            if node:
                node.maybe_tick(tick_time, self.graph)

