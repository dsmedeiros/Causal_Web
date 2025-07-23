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
    QGraphicsRectItem,
    QGraphicsItem,
    QGraphicsLineItem,
    QGraphicsScene,
    QGraphicsView,
    QMenu,
)


from ..graph.model import GraphModel
from ..gui.state import (
    set_selected_node,
    set_selected_connection,
    set_selected_observer,
)
from ..command_stack import (
    CommandStack,
    MoveNodeCommand,
    MoveMetaNodeCommand,
    MoveObserverCommand,
    AddNodeCommand,
    AddObserverCommand,
    AddMetaNodeCommand,
    DeleteEdgeCommand,
    DeleteNodeCommand,
    DeleteObserverCommand,
    DeleteMetaNodeCommand,
)


def make_dashed_line(x1: float, y1: float, x2: float, y2: float) -> QGraphicsLineItem:
    """Return a dashed line item connecting ``(x1, y1)`` and ``(x2, y2)``."""
    line = QGraphicsLineItem()
    pen = QPen(Qt.darkGray)
    pen.setStyle(Qt.DotLine)
    line.setPen(pen)
    line.setLine(x1, y1, x2, y2)
    line.setZValue(0)
    return line


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
        # ensure itemChange is triggered so connected edges update while moving
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges)
        if canvas.editable:
            self.setFlag(QGraphicsItem.ItemIsMovable)
            self.setFlag(QGraphicsItem.ItemIsSelectable)
        self.setZValue(1)
        self.edges: list[EdgeItem] = []
        self._drag_start: Optional[QPointF] = None

    def itemChange(self, change, value):  # type: ignore[override]
        """Redraw connected edges whenever the node moves."""
        if change == QGraphicsItem.ItemPositionHasChanged:
            for edge in self.edges:
                edge.update_position()
            self.canvas.update_meta_lines_for_node(self.node_id)
            self.canvas.update_observer_lines_for_node(self.node_id)
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


class MetaNodeItem(QGraphicsEllipseItem):
    """Ellipse representing a meta node with dashed member links."""

    def __init__(
        self,
        meta_id: str,
        x: float,
        y: float,
        members: list[str],
        canvas: "CanvasWidget",
        radius: float = 30.0,
    ) -> None:
        super().__init__(-radius, -radius, radius * 2, radius * 2)
        self.meta_id = meta_id
        self.canvas = canvas
        self.members = members
        self.setPos(QPointF(x, y))
        pen = QPen(Qt.darkGray)
        pen.setStyle(Qt.DotLine)
        self.setPen(pen)
        self.setBrush(QBrush(Qt.lightGray))
        self.setZValue(0.5)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges)
        if canvas.editable:
            self.setFlag(QGraphicsItem.ItemIsMovable)
            self.setFlag(QGraphicsItem.ItemIsSelectable)
        self.lines: list[QGraphicsLineItem] = []
        self._drag_start: Optional[QPointF] = None
        self.update_lines()

    def itemChange(self, change, value):  # type: ignore[override]
        if change == QGraphicsItem.ItemPositionHasChanged:
            self.update_lines()
            self.canvas.update_meta_lines_for_meta(self.meta_id)
        return super().itemChange(change, value)

    def update_lines(self) -> None:
        scene = self.scene()
        if scene is None:
            return
        for line in self.lines:
            scene.removeItem(line)
        self.lines.clear()
        for nid in self.members:
            node = self.canvas.nodes.get(nid)
            if node:
                line = make_dashed_line(self.x(), self.y(), node.x(), node.y())
                scene.addItem(line)
                self.lines.append(line)

    def mousePressEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        if event.button() == Qt.LeftButton:
            self.canvas.meta_node_selected.emit(self.meta_id)
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
                self.canvas.meta_node_moved(self.meta_id, self._drag_start, self.pos())
            self._drag_start = None
        super().mouseReleaseEvent(event)


class ObserverItem(QGraphicsRectItem):
    """Square item representing an observer with dotted target links."""

    def __init__(
        self,
        index: int,
        x: float,
        y: float,
        targets: list[str] | None,
        canvas: "CanvasWidget",
        size: float = 40.0,
    ) -> None:
        super().__init__(-size / 2, -size / 2, size, size)
        self.index = index
        self.canvas = canvas
        self.targets = targets or []
        self.setPos(QPointF(x, y))
        pen = QPen(Qt.darkGray)
        self.setPen(pen)
        self.setBrush(QBrush(Qt.white))
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges)
        if canvas.editable:
            self.setFlag(QGraphicsItem.ItemIsMovable)
            self.setFlag(QGraphicsItem.ItemIsSelectable)
        self.setZValue(1)
        self.lines: list[QGraphicsLineItem] = []
        self._drag_start: Optional[QPointF] = None
        self.update_lines()

    def itemChange(self, change, value):  # type: ignore[override]
        if change == QGraphicsItem.ItemPositionHasChanged:
            self.update_lines()
        return super().itemChange(change, value)

    def update_lines(self) -> None:
        scene = self.scene()
        if scene is None:
            return
        for line in self.lines:
            scene.removeItem(line)
        self.lines.clear()
        targets = self.targets or list(self.canvas.nodes)
        for nid in targets:
            node = self.canvas.nodes.get(nid)
            if node:
                line = make_dashed_line(self.x(), self.y(), node.x(), node.y())
                scene.addItem(line)
                self.lines.append(line)

    def mousePressEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        if event.button() == Qt.LeftButton:
            set_selected_observer(self.index)
            self.canvas.observer_selected.emit(self.index)
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
                self.canvas.observer_moved(self.index, self._drag_start, self.pos())
            self._drag_start = None
        super().mouseReleaseEvent(event)


class CanvasWidget(QGraphicsView):
    """Graphics view displaying nodes and edges with antialiasing enabled."""

    node_selected = Signal(str)
    connection_request = Signal(str, str)
    connection_selected = Signal(str, int)
    node_position_changed = Signal(str, float, float)
    meta_node_selected = Signal(str)
    meta_node_position_changed = Signal(str, float, float)
    observer_selected = Signal(int)
    observer_position_changed = Signal(int, float, float)

    def __init__(
        self, parent: Optional[QGraphicsView] = None, *, editable: bool = True
    ):
        super().__init__(parent)
        self.editable = editable
        self.setScene(QGraphicsScene(self))
        self.setRenderHint(QPainter.Antialiasing)
        self.nodes: Dict[str, NodeItem] = {}
        self.meta_nodes: Dict[str, MetaNodeItem] = {}
        self.observers: Dict[int, ObserverItem] = {}
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
        self.meta_nodes.clear()
        self.observers.clear()

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

        for meta_id, data in model.meta_nodes.items():
            x, y = data.get("x", 0.0), data.get("y", 0.0)
            members = data.get("members", [])
            item = MetaNodeItem(meta_id, x, y, members, self)
            scene.addItem(item)
            item.update_lines()
            self.meta_nodes[meta_id] = item

        for idx, obs in enumerate(model.observers):
            x, y = obs.get("x", 0.0), obs.get("y", 0.0)
            targets = obs.get("target_nodes")
            item = ObserverItem(idx, x, y, targets, self)
            scene.addItem(item)
            item.update_lines()
            self.observers[idx] = item

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
        self.node_position_changed.emit(node_id, end.x(), end.y())

    def meta_node_moved(self, meta_id: str, start: QPointF, end: QPointF) -> None:
        if not self.editable:
            return
        cmd = MoveMetaNodeCommand(self.model, meta_id, (end.x(), end.y()))
        self.command_stack.do(cmd)
        self.meta_node_position_changed.emit(meta_id, end.x(), end.y())

    def observer_moved(self, index: int, start: QPointF, end: QPointF) -> None:
        if not self.editable:
            return
        cmd = MoveObserverCommand(self.model, index, (end.x(), end.y()))
        self.command_stack.do(cmd)
        self.observer_position_changed.emit(index, end.x(), end.y())

    def update_meta_lines_for_node(self, node_id: str) -> None:
        for meta in self.meta_nodes.values():
            if node_id in meta.members:
                meta.update_lines()

    def update_meta_lines_for_meta(self, meta_id: str) -> None:
        meta = self.meta_nodes.get(meta_id)
        if meta is None:
            return
        meta.update_lines()

    def update_observer_lines_for_node(self, node_id: str) -> None:
        for obs in self.observers.values():
            if not obs.targets or node_id in obs.targets:
                obs.update_lines()

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

    # ---- context menu and editing helpers -----------------------------------

    def contextMenuEvent(self, event) -> None:
        if not self.editable:
            return super().contextMenuEvent(event)

        item = self.itemAt(event.pos())
        menu = QMenu(self)
        scene_pos = self.mapToScene(event.pos())

        actions = {}
        if isinstance(item, NodeItem):
            actions[menu.addAction("Delete Node")] = lambda: self.delete_node(
                item.node_id
            )
        elif isinstance(item, EdgeItem):
            actions[menu.addAction("Delete Connection")] = (
                lambda: self.delete_connection(item.index, item.connection_type)
            )
        elif isinstance(item, ObserverItem):
            actions[menu.addAction("Delete Observer")] = lambda: self.delete_observer(
                item.index
            )
        elif isinstance(item, MetaNodeItem):
            actions[menu.addAction("Delete MetaNode")] = lambda: self.delete_meta_node(
                item.meta_id
            )
        else:
            actions[menu.addAction("Add Node")] = lambda: self.add_node_at(
                scene_pos.x(), scene_pos.y()
            )
            actions[menu.addAction("Add Observer")] = lambda: self.add_observer_at(
                scene_pos.x(), scene_pos.y()
            )
            actions[menu.addAction("Add MetaNode")] = lambda: self.add_meta_node_at(
                scene_pos.x(), scene_pos.y()
            )

        chosen = menu.exec(event.globalPos())
        if chosen in actions:
            actions[chosen]()

    # -- add helpers --

    def add_node_at(self, x: float, y: float) -> None:
        model = self.model
        idx = 1
        while f"N{idx}" in model.nodes:
            idx += 1
        cmd = AddNodeCommand(model, f"N{idx}", {"x": x, "y": y})
        self.command_stack.do(cmd)
        self.load_model(model)

    def add_observer_at(self, x: float, y: float) -> None:
        model = self.model
        idx = 1
        existing = {o.get("id") for o in model.observers}
        while f"OBS{idx}" in existing:
            idx += 1
        obs = {"id": f"OBS{idx}", "monitors": [], "frequency": 1.0, "x": x, "y": y}
        cmd = AddObserverCommand(model, obs)
        self.command_stack.do(cmd)
        self.load_model(model)

    def add_meta_node_at(self, x: float, y: float) -> None:
        model = self.model
        idx = 1
        while f"MN{idx}" in model.meta_nodes:
            idx += 1
        meta_id = f"MN{idx}"
        data = {
            "members": [],
            "constraints": {},
            "type": "Configured",
            "collapsed": False,
            "x": x,
            "y": y,
        }
        cmd = AddMetaNodeCommand(model, meta_id, data)
        self.command_stack.do(cmd)
        self.load_model(model)

    # -- delete helpers --

    def delete_connection(self, index: int, connection_type: str) -> None:
        cmd = DeleteEdgeCommand(self.model, index, connection_type)
        self.command_stack.do(cmd)
        self.load_model(self.model)

    def delete_node(self, node_id: str) -> None:
        cmd = DeleteNodeCommand(self.model, node_id)
        self.command_stack.do(cmd)
        self.load_model(self.model)

    def delete_observer(self, index: int) -> None:
        cmd = DeleteObserverCommand(self.model, index)
        self.command_stack.do(cmd)
        self.load_model(self.model)

    def delete_meta_node(self, meta_id: str) -> None:
        cmd = DeleteMetaNodeCommand(self.model, meta_id)
        self.command_stack.do(cmd)
        self.load_model(self.model)
