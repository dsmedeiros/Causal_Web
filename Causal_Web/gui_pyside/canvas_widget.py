from __future__ import annotations

"""Reusable :class:`QGraphicsView` for visualising :class:`GraphModel` graphs."""

from typing import Dict, Optional

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QBrush, QMouseEvent, QPen, QWheelEvent, QPainter
from PySide6.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsLineItem,
    QGraphicsScene,
    QGraphicsView,
)

from ..graph.model import GraphModel
from ..gui.state import set_selected_node


class NodeItem(QGraphicsEllipseItem):
    """Movable ellipse representing a graph node."""

    def __init__(self, node_id: str, x: float, y: float, radius: float = 20.0):
        super().__init__(-radius, -radius, radius * 2, radius * 2)
        self.node_id = node_id
        self.setPos(QPointF(x, y))
        self.setBrush(QBrush(Qt.gray))
        self.setPen(QPen(Qt.lightGray))
        self.setFlag(QGraphicsItem.ItemIsMovable)
        self.setFlag(QGraphicsItem.ItemIsSelectable)
        self.setZValue(1)
        self.edges: list[EdgeItem] = []

    def itemChange(self, change, value):  # type: ignore[override]
        if change == QGraphicsItem.ItemPositionChange:
            for edge in self.edges:
                edge.update_position()
        return super().itemChange(change, value)

    def mousePressEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        if event.button() == Qt.LeftButton:
            set_selected_node(self.node_id)
        super().mousePressEvent(event)


class EdgeItem(QGraphicsLineItem):
    """Line connecting two NodeItems."""

    def __init__(self, source: NodeItem, target: NodeItem):
        super().__init__()
        self.source = source
        self.target = target
        pen = QPen(Qt.darkGray)
        pen.setWidth(2)
        self.setPen(pen)
        self.setZValue(0)
        self.update_position()
        source.edges.append(self)
        target.edges.append(self)

    def update_position(self) -> None:
        self.setLine(self.source.x(), self.source.y(), self.target.x(), self.target.y())


class CanvasWidget(QGraphicsView):
    """Graphics view displaying nodes and edges with antialiasing enabled."""

    def __init__(self, parent: Optional[QGraphicsView] = None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.setRenderHint(QPainter.Antialiasing)
        self.nodes: Dict[str, NodeItem] = {}
        self._pan_start: Optional[QPointF] = None

    def load_model(self, model: GraphModel) -> None:
        """Populate the scene from ``model``."""

        scene = self.scene()
        if scene is None:
            return
        scene.clear()
        self.nodes.clear()

        for node_id, data in model.nodes.items():
            x, y = data.get("x", 0.0), data.get("y", 0.0)
            item = NodeItem(node_id, x, y)
            scene.addItem(item)
            self.nodes[node_id] = item

        for edge in model.edges:
            src = self.nodes.get(edge.get("from"))
            dst = self.nodes.get(edge.get("to"))
            if src and dst:
                scene.addItem(EdgeItem(src, dst))

    # ---- interaction -------------------------------------------------
    def wheelEvent(self, event: QWheelEvent) -> None:
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.scale(factor, factor)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MiddleButton:
            self._pan_start = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
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
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MiddleButton:
            self._pan_start = None
            self.setCursor(Qt.ArrowCursor)
        else:
            super().mouseReleaseEvent(event)
