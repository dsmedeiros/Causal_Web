"""File IO helpers for :mod:`Causal_Web.graph`."""

from __future__ import annotations

import json
from typing import Any

from .model import GraphModel


def load_graph(path: str) -> GraphModel:
    """Load a graph from ``path`` and return a :class:`GraphModel`."""
    with open(path) as f:
        data = json.load(f)
    _validate_graph(data)
    return GraphModel.from_dict(data)


def save_graph(path: str, graph: GraphModel) -> None:
    """Write ``graph`` to ``path`` in JSON format."""
    with open(path, "w") as f:
        json.dump(graph.to_dict(), f, indent=2)


def new_graph(starter_node: bool = False) -> GraphModel:
    """Return a new blank graph model."""
    return GraphModel.blank(starter_node)


def _validate_graph(data: dict[str, Any]) -> None:
    if "nodes" not in data or "edges" not in data:
        raise ValueError("Graph file must contain 'nodes' and 'edges'")
    if not isinstance(data["nodes"], (dict, list)):
        raise ValueError("'nodes' must be a dict or list")
    if not isinstance(data["edges"], list):
        raise ValueError("'edges' must be a list")
    for edge in data["edges"]:
        if not isinstance(edge, dict):
            raise ValueError("edge entries must be objects")
        if "from" not in edge or "to" not in edge:
            raise ValueError("edge missing 'from' or 'to'")
    if "observers" in data and not isinstance(data["observers"], list):
        raise ValueError("'observers' must be a list")
