from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class GraphModel:
    """In-memory representation of a graph for the GUI."""

    nodes: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    edges: List[Dict[str, Any]] = field(default_factory=list)
    bridges: List[Dict[str, Any]] = field(default_factory=list)
    tick_sources: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the model to a plain ``dict``."""
        return {
            "nodes": self.nodes,
            "edges": self.edges,
            "bridges": self.bridges,
            "tick_sources": self.tick_sources,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GraphModel":
        """Construct a :class:`GraphModel` from ``data``."""
        model = cls()
        nodes = data.get("nodes", {})
        if isinstance(nodes, list):
            model.nodes = {n.get("id", str(i)): n for i, n in enumerate(nodes)}
        else:
            model.nodes = dict(nodes)
        model.edges = list(data.get("edges", []))
        model.bridges = list(data.get("bridges", []))
        model.tick_sources = list(data.get("tick_sources", []))
        return model

    @classmethod
    def blank(cls, starter_node: bool = False) -> "GraphModel":
        """Return a new empty graph optionally populated with one node."""
        model = cls()
        if starter_node:
            model.nodes["N1"] = {
                "x": 0.0,
                "y": 0.0,
                "frequency": 1.0,
                "refractory_period": 2.0,
                "base_threshold": 0.5,
                "phase": 0.0,
                "origin_type": "seed",
                "generation_tick": 0,
                "parent_ids": [],
            }
        return model

    def node_position(self, node_id: str) -> tuple[float, float] | None:
        """Return the ``(x, y)`` position for ``node_id`` if present."""
        node = self.nodes.get(node_id)
        if node is None:
            return None
        return node.get("x", 0.0), node.get("y", 0.0)

    def add_node(
        self,
        node_id: str,
        *,
        x: float = 0.0,
        y: float = 0.0,
        frequency: float = 1.0,
        refractory_period: float = 2.0,
        base_threshold: float = 0.5,
    ) -> None:
        """Insert a new node into the model."""

        self.nodes[node_id] = {
            "x": x,
            "y": y,
            "frequency": frequency,
            "refractory_period": refractory_period,
            "base_threshold": base_threshold,
            "phase": 0.0,
            "origin_type": "seed",
            "generation_tick": 0,
            "parent_ids": [],
        }

    def get_edges(self) -> List[Dict[str, Any]]:
        """Return a list of edges in the graph."""
        return self.edges

    # ---- Connection management convenience methods ----

    def add_connection(
        self,
        source: str,
        target: str,
        *,
        delay: float = 1.0,
        attenuation: float = 1.0,
        connection_type: str = "edge",
        **props: Any,
    ) -> None:
        """Add an edge or bridge to the model.

        Parameters
        ----------
        source:
            Identifier of the source node.
        target:
            Identifier of the target node.
        delay:
            Propagation delay for the connection.
        attenuation:
            Attenuation factor for the connection.
        connection_type:
            ``"edge"`` for a directed edge or ``"bridge"`` for an undirected
            bridge.
        """

        if connection_type not in {"edge", "bridge"}:
            raise ValueError("connection_type must be 'edge' or 'bridge'")
        if source not in self.nodes or target not in self.nodes:
            raise ValueError("source and target must exist in the graph")
        if source == target:
            raise ValueError("self-loops are not allowed")

        if connection_type == "edge":
            if any(e["from"] == source and e["to"] == target for e in self.edges):
                raise ValueError("duplicate edge")
            record = {
                "from": source,
                "to": target,
                "delay": delay,
                "attenuation": attenuation,
            }
            record.update(props)
            self.edges.append(record)
        else:
            if any(set(b.get("nodes", [])) == {source, target} for b in self.bridges):
                raise ValueError("duplicate bridge")
            record = {
                "nodes": [source, target],
                "delay": delay,
                "attenuation": attenuation,
                "status": "active",
            }
            record.update(props)
            self.bridges.append(record)

    def update_connection(
        self,
        index: int,
        connection_type: str = "edge",
        **kwargs: Any,
    ) -> None:
        """Update properties of an existing connection."""

        target_list = self.edges if connection_type == "edge" else self.bridges
        if index < 0 or index >= len(target_list):
            raise IndexError("connection index out of range")
        target_list[index].update(kwargs)

    def remove_connection(self, index: int, connection_type: str = "edge") -> None:
        """Delete an edge or bridge from the model."""

        target_list = self.edges if connection_type == "edge" else self.bridges
        if index < 0 or index >= len(target_list):
            raise IndexError("connection index out of range")
        del target_list[index]

    def apply_spring_layout(self) -> None:
        """Position nodes using ``networkx.spring_layout``."""

        import networkx as nx

        g = nx.Graph()
        for node_id in self.nodes:
            g.add_node(node_id)
        for edge in self.edges:
            g.add_edge(edge["from"], edge["to"])
        for bridge in self.bridges:
            nodes = bridge.get("nodes", [])
            if len(nodes) == 2:
                g.add_edge(nodes[0], nodes[1])

        pos = nx.spring_layout(g)
        for nid, coords in pos.items():
            node = self.nodes.get(nid, {})
            node["x"], node["y"] = float(coords[0]), float(coords[1])
            self.nodes[nid] = node
