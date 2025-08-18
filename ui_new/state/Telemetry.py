from __future__ import annotations

"""Telemetry model exposed to QML panels."""

from PySide6.QtCore import QObject, Property, Signal

from telemetry import RollingTelemetry


class TelemetryModel(QObject):
    """Report node/edge counts and rolling telemetry histories."""

    nodeCountChanged = Signal(int)
    edgeCountChanged = Signal(int)
    countersChanged = Signal(dict)
    invariantsChanged = Signal(dict)
    depthChanged = Signal(int)
    depthLabelChanged = Signal(str)

    def __init__(self, max_points: int = 100) -> None:
        super().__init__()
        self._node_count = 0
        self._edge_count = 0
        self._telemetry = RollingTelemetry(max_points=max_points)
        self._depth = 0
        self._depth_label = "depth"

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
    def _get_counters(self) -> dict[str, list[float]]:
        return self._telemetry.get_counters()

    counters = Property(dict, _get_counters, notify=countersChanged)

    # ------------------------------------------------------------------
    def _get_invariants(self) -> dict[str, list[float]]:
        return self._telemetry.get_invariants()

    invariants = Property(dict, _get_invariants, notify=invariantsChanged)

    # ------------------------------------------------------------------
    def record(
        self,
        counters: dict[str, float] | None = None,
        invariants: dict[str, bool] | None = None,
        depth: int | None = None,
        label: str = "depth",
    ) -> None:
        """Record a telemetry sample and emit change signals."""
        if depth is not None and depth != self._depth:
            self._depth = depth
            self.depthChanged.emit(depth)
        if label != self._depth_label:
            self._depth_label = label
            self.depthLabelChanged.emit(label)
        self._telemetry.record(counters, invariants)
        self.countersChanged.emit(self._telemetry.get_counters())
        self.invariantsChanged.emit(self._telemetry.get_invariants())

    def update_counts(self, nodes: int, edges: int) -> None:
        """Convenience method to update both counts."""

        self._set_node_count(nodes)
        self._set_edge_count(edges)

    # ------------------------------------------------------------------
    def _get_depth(self) -> int:
        return self._depth

    depth = Property(int, _get_depth, notify=depthChanged)

    # ------------------------------------------------------------------
    def _get_depth_label(self) -> str:
        return self._depth_label

    depthLabel = Property(str, _get_depth_label, notify=depthLabelChanged)
