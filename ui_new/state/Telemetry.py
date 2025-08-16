from __future__ import annotations

"""Telemetry model exposed to QML panels."""

from PySide6.QtCore import QObject, Property, Signal


class TelemetryModel(QObject):
    """Simple model reporting node and edge counts."""

    nodeCountChanged = Signal(int)
    edgeCountChanged = Signal(int)

    def __init__(self) -> None:
        super().__init__()
        self._node_count = 0
        self._edge_count = 0

    # ------------------------------------------------------------------
    def _get_node_count(self) -> int:
        return self._node_count

    def _set_node_count(self, value: int) -> None:
        if self._node_count != value:
            self._node_count = value
            self.nodeCountChanged.emit(value)

    nodeCount = Property(int, _get_node_count, _set_node_count, notify=nodeCountChanged)

    # ------------------------------------------------------------------
    def _get_edge_count(self) -> int:
        return self._edge_count

    def _set_edge_count(self, value: int) -> None:
        if self._edge_count != value:
            self._edge_count = value
            self.edgeCountChanged.emit(value)

    edgeCount = Property(int, _get_edge_count, _set_edge_count, notify=edgeCountChanged)

    # ------------------------------------------------------------------
    def update_counts(self, nodes: int, edges: int) -> None:
        """Convenience method to update both counts."""
        self._set_node_count(nodes)
        self._set_edge_count(edges)
