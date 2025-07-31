from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from .types import (
    BridgeData,
    EdgeData,
    GraphDict,
    MetaNodeData,
    NodeData,
    ObserverData,
)


@dataclass
class GraphModel:
    """In-memory representation of a graph for the GUI."""

    nodes: Dict[str, NodeData] = field(default_factory=dict)
    edges: List[EdgeData] = field(default_factory=list)
    bridges: List[BridgeData] = field(default_factory=list)
    tick_sources: List[Dict[str, Any]] = field(default_factory=list)
    observers: List[ObserverData] = field(default_factory=list)
    meta_nodes: Dict[str, MetaNodeData] = field(default_factory=dict)

    def to_dict(self) -> GraphDict:
        """Serialize the model to a plain ``dict`` suitable for JSON."""
        return {
            "nodes": self.nodes,
            "edges": self.edges,
            "bridges": self.bridges,
            "tick_sources": self.tick_sources,
            "observers": self.observers,
            "meta_nodes": self.meta_nodes,
        }

    @classmethod
    def from_dict(cls, data: GraphDict) -> "GraphModel":
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
        model.observers = list(data.get("observers", []))
        for obs in model.observers:
            obs.setdefault("x", 0.0)
            obs.setdefault("y", 0.0)
        model.meta_nodes = dict(data.get("meta_nodes", {}))
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
                "allow_self_connection": False,
            }
        model.meta_nodes = {}
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
        allow_self_connection: bool = False,
    ) -> None:
        """Insert a new node into the model."""

        self.nodes[node_id] = NodeData(
            id=node_id,
            x=x,
            y=y,
            frequency=frequency,
            refractory_period=refractory_period,
            base_threshold=base_threshold,
            phase=0.0,
            origin_type="seed",
            generation_tick=0,
            parent_ids=[],
            allow_self_connection=allow_self_connection,
        )

    def get_edges(self) -> List[EdgeData]:
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
            if connection_type != "edge" or not self.nodes[source].get(
                "allow_self_connection", False
            ):
                raise ValueError("self-loops are not allowed")

        if connection_type == "edge":
            if any(e["from"] == source and e["to"] == target for e in self.edges):
                raise ValueError("duplicate edge")
            record: EdgeData = {
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
            record: BridgeData = {
                "nodes": [source, target],
                "delay": delay,
                "attenuation": attenuation,
                "status": "active",
            }
            if "is_entangled" in props and props["is_entangled"]:
                record["is_entangled"] = True
                record["entangled_id"] = props.get("entangled_id")
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

    # ---- Meta node management -------------------------------------------------

    def add_meta_node(
        self,
        meta_id: str,
        *,
        members: List[str] | None = None,
        constraints: Dict[str, Any] | None = None,
        type: str = "Configured",
        origin: str | None = None,
        collapsed: bool = False,
        x: float = 0.0,
        y: float = 0.0,
    ) -> None:
        """Insert a new meta node definition."""

        data: MetaNodeData = {
            "members": list(members or []),
            "constraints": constraints or {},
            "type": type,
            "collapsed": collapsed,
            "x": x,
            "y": y,
        }
        if origin is not None:
            data["origin"] = origin
        self.meta_nodes[meta_id] = data

    # ---- Observer management -------------------------------------------------

    def add_observer(
        self,
        obs_id: str,
        *,
        monitors: List[str] | None = None,
        frequency: float = 1.0,
        target_nodes: List[str] | None = None,
        x: float = 0.0,
        y: float = 0.0,
    ) -> None:
        """Insert a new observer definition with optional position."""

        data: ObserverData = {
            "id": obs_id,
            "monitors": monitors or [],
            "frequency": frequency,
            "x": x,
            "y": y,
        }
        if target_nodes:
            data["target_nodes"] = list(target_nodes)
        self.observers.append(data)

    # ---- Removal helpers -----------------------------------------------------

    def remove_node(self, node_id: str) -> None:
        """Delete ``node_id`` and any references to it."""

        if node_id not in self.nodes:
            return
        self.nodes.pop(node_id)

        self.edges = [
            e for e in self.edges if e.get("from") != node_id and e.get("to") != node_id
        ]
        self.bridges = [b for b in self.bridges if node_id not in b.get("nodes", [])]

        for obs in self.observers:
            obs["monitors"] = [n for n in obs.get("monitors", []) if n != node_id]
            if "target_nodes" in obs:
                obs["target_nodes"] = [n for n in obs["target_nodes"] if n != node_id]

        for meta in self.meta_nodes.values():
            if node_id in meta.get("members", []):
                meta["members"] = [n for n in meta["members"] if n != node_id]

    def remove_observer(self, index: int) -> None:
        """Remove the observer at ``index`` if valid."""

        if 0 <= index < len(self.observers):
            self.observers.pop(index)

    def remove_meta_node(self, meta_id: str) -> None:
        """Remove ``meta_id`` from the model."""

        self.meta_nodes.pop(meta_id, None)

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
