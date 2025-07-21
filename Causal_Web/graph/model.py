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
