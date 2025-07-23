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


_active_file: str | None = None
_selected_node: str | None = None
_selected_connection: tuple[str, int] | None = None
_selected_observer: int | None = None
_graph_dirty: bool = False


def get_active_file() -> str | None:
    """Return the path of the currently loaded graph file."""
    return _active_file


def set_active_file(path: str | None) -> None:
    """Record the path of the loaded graph file."""
    global _active_file
    _active_file = path
    clear_graph_dirty()


def get_selected_node() -> str | None:
    """Return the currently selected node id, if any."""
    return _selected_node


def set_selected_node(node_id: str | None) -> None:
    """Update the currently selected node id."""
    global _selected_node
    _selected_node = node_id


def get_selected_connection() -> tuple[str, int] | None:
    """Return (type, index) of the selected connection if any."""
    return _selected_connection


def set_selected_connection(conn: tuple[str, int] | None) -> None:
    """Update the selected connection reference."""
    global _selected_connection
    _selected_connection = conn


def get_selected_observer() -> int | None:
    """Return the index of the selected observer, if any."""
    return _selected_observer


def set_selected_observer(index: int | None) -> None:
    """Update the currently selected observer index."""
    global _selected_observer
    _selected_observer = index


def mark_graph_dirty() -> None:
    """Flag the active graph as modified since the last load/save."""
    global _graph_dirty
    _graph_dirty = True


def clear_graph_dirty() -> None:
    """Reset the dirty flag for the active graph."""
    global _graph_dirty
    _graph_dirty = False


def is_graph_dirty() -> bool:
    """Return ``True`` if the active graph has unapplied changes."""
    return _graph_dirty
