from __future__ import annotations

"""Reusable :class:`QGraphicsView` for visualising :class:`GraphModel` graphs."""

from typing import Any, Dict, Optional, Tuple

from PySide6.QtCore import QPoint, QPointF, Qt, Signal, QTimer
from PySide6.QtGui import (
    QBrush,
    QMouseEvent,
    QPen,
    QWheelEvent,
    QPainter,
    QPainterPath,
    QContextMenuEvent,
)
from PySide6.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsRectItem,
    QGraphicsItem,
    QGraphicsLineItem,
    QGraphicsPathItem,
    QGraphicsScene,
    QGraphicsView,
    QGraphicsSimpleTextItem,
    QMenu,
)


from ..graph.model import GraphModel
from ..gui.state import (
    set_selected_node,
    set_selected_connection,
    set_selected_observer,
)
from ..gui.command_stack import (
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
from ..view import ViewSnapshot


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

    def itemChange(
        self,
        change: QGraphicsItem.GraphicsItemChange,
        value: Any,
    ) -> Any:  # type: ignore[override]
        """Redraw connected edges whenever the node moves."""
        if change == QGraphicsItem.ItemPositionHasChanged:
            for edge in self.edges:
                edge.update_position()
            self.canvas.update_meta_lines_for_node(self.node_id)
            self.canvas.update_observer_lines_for_node(self.node_id)
        return super().itemChange(change, value)

    def mousePressEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        if event.button() == Qt.LeftButton:
            if not self.canvas._connect_mode:
                set_selected_node(self.node_id)
                self.canvas.node_selected.emit(self.node_id)
                if self.canvas.editable:
                    self._drag_start = self.pos()
            else:
                self._drag_start = None
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


class SelfEdgeItem(QGraphicsPathItem):
    """Curved edge originating and ending on the same node."""

    def __init__(self, node: NodeItem, canvas: "CanvasWidget", index: int) -> None:
        super().__init__()
        self.node = node
        self.canvas = canvas
        self.index = index
        pen = QPen(Qt.darkGray)
        pen.setWidth(2)
        self.setPen(pen)
        self.setZValue(0)
        if canvas.editable:
            self.setFlag(QGraphicsItem.ItemIsSelectable)
        node.edges.append(self)
        self.update_position()

    def mousePressEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        if event.button() == Qt.LeftButton:
            set_selected_connection(("edge", self.index))
            self.canvas.connection_selected.emit("edge", self.index)
        super().mousePressEvent(event)

    def update_position(self) -> None:
        radius = self.node.rect().width() / 2
        path = QPainterPath()
        start = QPointF(self.node.x() + radius, self.node.y())
        end = QPointF(self.node.x() - radius, self.node.y())
        offset = radius * 2
        ctrl1 = QPointF(self.node.x() + offset, self.node.y() - offset)
        ctrl2 = QPointF(self.node.x() - offset, self.node.y() - offset)
        path.moveTo(start)
        path.cubicTo(ctrl1, ctrl2, end)
        self.setPath(path)


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

    def itemChange(
        self,
        change: QGraphicsItem.GraphicsItemChange,
        value: Any,
    ) -> Any:  # type: ignore[override]
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

    def itemChange(
        self,
        change: QGraphicsItem.GraphicsItemChange,
        value: Any,
    ) -> Any:  # type: ignore[override]
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
    ) -> None:
        """Initialize the canvas widget."""
        super().__init__(parent)
        self.editable = editable
        self.setScene(QGraphicsScene(self))
        self.setRenderHint(QPainter.Antialiasing)
        self.nodes: Dict[str, NodeItem] = {}
        self.edges: Dict[tuple[str, str], EdgeItem | SelfEdgeItem] = {}
        self.meta_nodes: Dict[str, MetaNodeItem] = {}
        self.observers: Dict[int, ObserverItem] = {}
        self.model: GraphModel = GraphModel.blank()
        self._pan_start: Optional[QPointF] = None
        self.command_stack = CommandStack()
        self._connect_mode: bool = False
        self._connect_start: Optional[NodeItem] = None
        self._connect_dragged: bool = False
        self._temp_edge: Optional[QGraphicsLineItem] = None
        self._hud_item: Optional[QGraphicsSimpleTextItem] = None
        if not self.editable:
            scene = self.scene()
            if scene is not None:
                self._hud_item = QGraphicsSimpleTextItem("")
                self._hud_item.setBrush(QBrush(Qt.white))
                self._hud_item.setZValue(2)
                self._hud_item.setPos(5, 5)
                scene.addItem(self._hud_item)

    def load_model(self, model: GraphModel) -> None:
        """Populate the scene from ``model``."""

        self.model = model
        scene = self.scene()
        if scene is None:
            return
        scene.clear()
        self.nodes.clear()
        self.edges.clear()
        self.meta_nodes.clear()
        self.observers.clear()
        if self._hud_item is not None:
            scene.addItem(self._hud_item)
            self._hud_item.setPos(5, 5)

        for node_id, data in model.nodes.items():
            x, y = data.get("x", 0.0), data.get("y", 0.0)
            item = NodeItem(node_id, x, y, self)
            scene.addItem(item)
            self.nodes[node_id] = item

        for idx, edge in enumerate(model.edges):
            src_id = edge.get("from")
            dst_id = edge.get("to")
            src = self.nodes.get(src_id)
            dst = self.nodes.get(dst_id)
            if src and dst:
                if src is dst:
                    item = SelfEdgeItem(src, self, idx)
                else:
                    item = EdgeItem(src, dst, self, idx, "edge")
                scene.addItem(item)
                self.edges[(src_id, dst_id)] = item

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

    def apply_diff(self, snapshot: ViewSnapshot) -> None:
        """Apply incremental updates from ``snapshot`` to the scene.

        Nodes and edges referenced in the snapshot are refreshed in-place using
        cached ``NodeItem`` and ``EdgeItem`` instances. Items are created or
        removed if the underlying :class:`GraphModel` now contains or lacks the
        corresponding entries.
        """

        scene = self.scene()
        if scene is None:
            return

        for view in snapshot.changed_nodes:
            node_id = view.id
            data = self.model.nodes.get(node_id)
            item = self.nodes.get(node_id)
            if data is None:
                if item is not None:
                    for edge in list(item.edges):
                        if isinstance(edge, SelfEdgeItem):
                            key = (node_id, node_id)
                            self.edges.pop(key, None)
                        else:
                            key = (edge.source.node_id, edge.target.node_id)
                            self.edges.pop(key, None)
                            edge.source.edges.remove(edge)
                            if edge.target is not edge.source:
                                edge.target.edges.remove(edge)
                        scene.removeItem(edge)
                    scene.removeItem(item)
                    self.nodes.pop(node_id, None)
                    self.update_meta_lines_for_node(node_id)
                    self.update_observer_lines_for_node(node_id)
                continue

            x, y = data.get("x", 0.0), data.get("y", 0.0)
            if item is None:
                new_item = NodeItem(node_id, x, y, self)
                scene.addItem(new_item)
                self.nodes[node_id] = new_item
                item = new_item
            else:
                item.setPos(QPointF(x, y))
            self.update_meta_lines_for_node(node_id)
            self.update_observer_lines_for_node(node_id)

        for view in snapshot.changed_edges:
            key = (view.src, view.dst)
            data = next(
                (
                    e
                    for e in self.model.edges
                    if e.get("from") == view.src and e.get("to") == view.dst
                ),
                None,
            )
            item = self.edges.get(key)
            if data is None:
                if item is not None:
                    if isinstance(item, SelfEdgeItem):
                        node = item.node
                        node.edges.remove(item)
                    else:
                        item.source.edges.remove(item)
                        item.target.edges.remove(item)
                    scene.removeItem(item)
                    self.edges.pop(key, None)
                continue

            src = self.nodes.get(view.src)
            dst = self.nodes.get(view.dst)
            if src is None or dst is None:
                continue
            idx = next(
                (
                    i
                    for i, e in enumerate(self.model.edges)
                    if e.get("from") == view.src and e.get("to") == view.dst
                ),
                -1,
            )
            if item is None:
                if src is dst:
                    new_edge = SelfEdgeItem(src, self, idx)
                else:
                    new_edge = EdgeItem(src, dst, self, idx, "edge")
                scene.addItem(new_edge)
                self.edges[key] = new_edge
            else:
                item.index = idx
                item.update_position()

    def update_hud(self, tick: int, depth: int, window: int) -> None:
        """Update the on-canvas HUD text."""

        if self._hud_item is not None:
            self._hud_item.setText(
                f"arrival-depth {tick} | depth {depth} | depth limit {window}"
            )

    def highlight_closed_windows(self, events: list) -> None:
        """Temporarily highlight nodes referenced by closed window events."""
        for ev in events:
            node_id = str(getattr(ev, "window_idx", ""))
            item = self.nodes.get(node_id)
            if item is None:
                continue
            original = item.brush()
            item.setBrush(QBrush(Qt.red))
            QTimer.singleShot(500, lambda it=item, br=original: it.setBrush(br))

    # ---- interaction -------------------------------------------------
    def wheelEvent(self, event: QWheelEvent) -> None:
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.scale(factor, factor)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MiddleButton or (
            event.button() == Qt.LeftButton
            and self.itemAt(event.pos()) is None
            and not self._connect_mode
        ):
            self._pan_start = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
        else:
            if not self.editable:
                super().mousePressEvent(event)
                return
            item = self.itemAt(event.pos())
            if self._connect_mode and isinstance(item, NodeItem):
                if self._connect_start is None:
                    self._connect_start = item
                    self._connect_dragged = False
                    if scene := self.scene():
                        pen = QPen(Qt.darkGray)
                        pen.setStyle(Qt.DashLine)
                        self._temp_edge = QGraphicsLineItem()
                        self._temp_edge.setPen(pen)
                        scene.addItem(self._temp_edge)
                        self._update_temp_edge(event.pos())
                else:
                    self._finalize_connection(item, event.pos())
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
                if event.buttons() & Qt.LeftButton:
                    self._connect_dragged = True
                self._update_temp_edge(event.pos())
            else:
                super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if (
            event.button() in (Qt.MiddleButton, Qt.LeftButton)
            and self._pan_start is not None
        ):
            self._pan_start = None
            self.setCursor(Qt.ArrowCursor)
        else:
            if self.editable and self._connect_start:
                item = self.itemAt(event.pos())
                if self._connect_dragged and isinstance(item, NodeItem):
                    self._finalize_connection(item, event.pos())
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

    def _finalize_connection(self, item: NodeItem, view_pos: QPoint) -> None:
        """Create a connection from ``_connect_start`` to ``item`` if allowed."""
        if item is self._connect_start:
            data = self.model.nodes.get(item.node_id, {})
            if data.get("allow_self_connection", False):
                self.connection_request.emit(item.node_id, item.node_id)
            else:
                self._show_status_message("Self-connection disabled", view_pos)
        else:
            self.connection_request.emit(self._connect_start.node_id, item.node_id)
        if self._temp_edge and self.scene():
            self.scene().removeItem(self._temp_edge)
        self._temp_edge = None
        self._connect_start = None
        self._connect_mode = False
        self._connect_dragged = False

    def _show_status_message(self, text: str, view_pos: QPoint) -> None:
        if scene := self.scene():
            item = QGraphicsSimpleTextItem(text)
            item.setBrush(QBrush(Qt.red))
            item.setZValue(2)
            scene_pos = self.mapToScene(view_pos)
            item.setPos(scene_pos)
            scene.addItem(item)
            QTimer.singleShot(1500, lambda: scene.removeItem(item))

    def enable_connection_mode(self) -> None:
        """Begin interactive connection creation."""
        if not self.editable:
            return
        self._connect_mode = True
        self._connect_start = None
        self._connect_dragged = False
        if self._temp_edge and self.scene():
            self.scene().removeItem(self._temp_edge)
        self._temp_edge = None

    def cancel_connection_mode(self) -> None:
        """Abort interactive connection mode and clear temporary state."""
        if self._temp_edge and self.scene():
            self.scene().removeItem(self._temp_edge)
        self._temp_edge = None
        self._connect_start = None
        self._connect_dragged = False
        self._connect_mode = False

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

    def contextMenuEvent(self, event: QContextMenuEvent) -> None:
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
