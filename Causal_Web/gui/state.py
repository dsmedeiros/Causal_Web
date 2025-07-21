"""Shared GUI state including the active :class:`GraphModel`."""

from __future__ import annotations

from ..graph.model import GraphModel

_active_graph: GraphModel | None = None


def get_graph() -> GraphModel:
    """Return the current graph, creating a blank one if needed."""
    global _active_graph
    if _active_graph is None:
        _active_graph = GraphModel.blank()
    return _active_graph


def set_graph(graph: GraphModel) -> None:
    """Replace the active graph."""
    global _active_graph
    _active_graph = graph
