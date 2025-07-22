from __future__ import annotations

"""Reusable :class:`QGraphicsView` for visualising :class:`GraphModel` graphs."""

from typing import Dict, Optional, Tuple

from PySide6.QtCore import QPoint, QPointF, Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QMouseEvent,
    QPen,
    QWheelEvent,
    QPainter,
)
from PySide6.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsLineItem,
    QGraphicsScene,
    QGraphicsView,
)

from ..graph.model import GraphModel
from ..gui.state import set_selected_node, set_selected_connection
from ..command_stack import CommandStack, MoveNodeCommand


class NodeItem(QGraphicsEllipseItem):
    """Movable ellipse representing a graph node."""

    def __init__(
        self,
        node_id: str,
        x: float,
        y: float,
        canvas: "CanvasWidget",
        radius: float = 20.0,
    ):
        super().__init__(-radius, -radius, radius * 2, radius * 2)
        self.node_id = node_id
        self.canvas = canvas
        self.setPos(QPointF(x, y))
        self.setBrush(QBrush(Qt.gray))
        self.setPen(QPen(Qt.lightGray))
        if canvas.editable:
            self.setFlag(QGraphicsItem.ItemIsMovable)
            self.setFlag(QGraphicsItem.ItemIsSelectable)
        self.setZValue(1)
        self.edges: list[EdgeItem] = []
        self._drag_start: Optional[QPointF] = None

    def itemChange(self, change, value):  # type: ignore[override]
        if change == QGraphicsItem.ItemPositionChange:
            for edge in self.edges:
                edge.update_position()
        return super().itemChange(change, value)

    def mousePressEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        if event.button() == Qt.LeftButton:
            set_selected_node(self.node_id)
            self.canvas.node_selected.emit(self.node_id)
            if self.canvas.editable:
                self._drag_start = self.pos()
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        if (
            event.button() == Qt.LeftButton
            and self._drag_start is not None
            and self.canvas.editable
        ):
            if self.pos() != self._drag_start:
                self.canvas.node_moved(self.node_id, self._drag_start, self.pos())
            self._drag_start = None
        super().mouseReleaseEvent(event)


class EdgeItem(QGraphicsLineItem):
    """Line connecting two NodeItems."""

    def __init__(
        self,
        source: NodeItem,
        target: NodeItem,
        canvas: "CanvasWidget",
        index: int,
        connection_type: str = "edge",
    ):
        super().__init__()
        self.source = source
        self.target = target
        self.canvas = canvas
        self.index = index
        self.connection_type = connection_type
        pen = QPen(Qt.darkGray)
        pen.setWidth(2)
        self.setPen(pen)
        self.setZValue(0)
        self.update_position()
        if canvas.editable:
            self.setFlag(QGraphicsItem.ItemIsSelectable)
        source.edges.append(self)
        target.edges.append(self)

    def mousePressEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        if event.button() == Qt.LeftButton:
            set_selected_connection((self.connection_type, self.index))
            self.canvas.connection_selected.emit(self.connection_type, self.index)
        super().mousePressEvent(event)

    def update_position(self) -> None:
        self.setLine(self.source.x(), self.source.y(), self.target.x(), self.target.y())


class CanvasWidget(QGraphicsView):
    """Graphics view displaying nodes and edges with antialiasing enabled."""

    node_selected = Signal(str)
    connection_request = Signal(str, str)
    connection_selected = Signal(str, int)

    def __init__(
        self, parent: Optional[QGraphicsView] = None, *, editable: bool = True
    ):
        super().__init__(parent)
        self.editable = editable
        self.setScene(QGraphicsScene(self))
        self.setRenderHint(QPainter.Antialiasing)
        self.nodes: Dict[str, NodeItem] = {}
        self.model: GraphModel = GraphModel.blank()
        self._pan_start: Optional[QPointF] = None
        self.command_stack = CommandStack()
        self._connect_mode: bool = False
        self._connect_start: Optional[NodeItem] = None
        self._temp_edge: Optional[QGraphicsLineItem] = None

    def load_model(self, model: GraphModel) -> None:
        """Populate the scene from ``model``."""

        self.model = model
        scene = self.scene()
        if scene is None:
            return
        scene.clear()
        self.nodes.clear()

        for node_id, data in model.nodes.items():
            x, y = data.get("x", 0.0), data.get("y", 0.0)
            item = NodeItem(node_id, x, y, self)
            scene.addItem(item)
            self.nodes[node_id] = item

        for idx, edge in enumerate(model.edges):
            src = self.nodes.get(edge.get("from"))
            dst = self.nodes.get(edge.get("to"))
            if src and dst:
                scene.addItem(EdgeItem(src, dst, self, idx, "edge"))

    # ---- interaction -------------------------------------------------
    def wheelEvent(self, event: QWheelEvent) -> None:
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.scale(factor, factor)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MiddleButton:
            self._pan_start = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
        else:
            if not self.editable:
                super().mousePressEvent(event)
                return
            item = self.itemAt(event.pos())
            if (
                self._connect_mode
                and self._connect_start is None
                and isinstance(item, NodeItem)
            ):
                self._connect_start = item
                if scene := self.scene():
                    pen = QPen(Qt.darkGray)
                    pen.setStyle(Qt.DashLine)
                    self._temp_edge = QGraphicsLineItem()
                    self._temp_edge.setPen(pen)
                    scene.addItem(self._temp_edge)
                    self._update_temp_edge(event.pos())
            else:
                super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._pan_start is not None:
            delta = event.pos() - self._pan_start
            self._pan_start = event.pos()
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - int(delta.x())
            )
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - int(delta.y())
            )
        else:
            if self.editable and self._connect_start and self._temp_edge is not None:
                self._update_temp_edge(event.pos())
            else:
                super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MiddleButton:
            self._pan_start = None
            self.setCursor(Qt.ArrowCursor)
        else:
            if self.editable and self._connect_start:
                item = self.itemAt(event.pos())
                if isinstance(item, NodeItem) and item is not self._connect_start:
                    self.connection_request.emit(
                        self._connect_start.node_id, item.node_id
                    )
                if self._temp_edge and self.scene():
                    self.scene().removeItem(self._temp_edge)
                self._temp_edge = None
                self._connect_start = None
                self._connect_mode = False
            else:
                super().mouseReleaseEvent(event)

    # ---- helpers ----------------------------------------------------
    def _update_temp_edge(self, view_pos: QPoint) -> None:
        """Update the dashed preview edge while dragging."""
        if self._temp_edge and self._connect_start:
            scene_pos = self.mapToScene(view_pos)
            self._temp_edge.setLine(
                self._connect_start.x(),
                self._connect_start.y(),
                scene_pos.x(),
                scene_pos.y(),
            )

    def enable_connection_mode(self) -> None:
        """Begin interactive connection creation."""
        if not self.editable:
            return
        self._connect_mode = True
        self._connect_start = None
        if self._temp_edge and self.scene():
            self.scene().removeItem(self._temp_edge)
        self._temp_edge = None

    def node_moved(self, node_id: str, start: QPointF, end: QPointF) -> None:
        if not self.editable:
            return
        cmd = MoveNodeCommand(self.model, node_id, (end.x(), end.y()))
        self.command_stack.do(cmd)

    def undo(self) -> None:
        if not self.editable:
            return
        self.command_stack.undo()
        self.load_model(self.model)

    def redo(self) -> None:
        if not self.editable:
            return
        self.command_stack.redo()
        self.load_model(self.model)

    def auto_layout(self) -> None:
        """Apply a spring layout to ``self.model`` and refresh the scene."""
        if not self.editable:
            return
        self.model.apply_spring_layout()
        self.load_model(self.model)
