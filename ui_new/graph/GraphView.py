from __future__ import annotations

"""Scene-graph renderer for causal graphs using Qt Quick instancing."""

from typing import Dict, Iterable, List, Tuple, Set

from PySide6.QtGui import QColor, QVector2D, QVector4D, QWheelEvent
from PySide6.QtQml import QmlElement
from PySide6.QtQuick import (
    QQuickItem,
    QSGGeometry,
    QSGGeometryNode,
    QSGMaterial,
    QSGMaterialShader,
    QSGMaterialType,
    QSGRendererInterface,
    QSGNode,
)
from PySide6.QtCore import QByteArray, Property, QRectF, Signal, Slot

QML_IMPORT_NAME = "CausalGraph"
QML_IMPORT_MAJOR_VERSION = 1


class _InstancedMaterial(QSGMaterial):
    """Flat material feeding per-instance attributes to the shader."""

    _TYPE = QSGMaterialType()

    def __init__(self) -> None:
        super().__init__()
        # Attribute buffers mirroring geometry instance data
        self.offsets: List[QVector2D] = []
        self.colors: List[QVector4D] = []
        self.flags: List[float] = []

    def type(self) -> QSGMaterialType:  # pragma: no cover - Qt binding detail
        return self._TYPE

    def createShader(
        self, render_mode: QSGRendererInterface.RenderMode
    ) -> QSGMaterialShader:  # pragma: no cover - Qt binding detail
        return _InstancedShader()


class _InstancedShader(QSGMaterialShader):
    """Shader consuming per-instance attributes for quads."""

    VERTEX = QByteArray(
        b"""
        attribute highp vec4 aVertex;
        attribute highp vec2 aOffset;
        attribute lowp vec4 aColor;
        attribute float aFlag;
        varying lowp vec4 vColor;
        varying float vFlag;
        void main(){
            vColor = aColor;
            vFlag = aFlag;
            gl_Position = aVertex + vec4(aOffset, 0.0, 0.0);
        }
        """
    )

    FRAG = QByteArray(
        b"""
        varying lowp vec4 vColor;
        varying float vFlag;
        void main(){
            if (vFlag < 0.5) discard;
            gl_FragColor = vColor;
        }
        """
    )

    def __init__(self) -> None:  # pragma: no cover - Qt binding detail
        super().__init__()
        if hasattr(self, "setShaderSourceCode"):
            self.setShaderSourceCode(QSGMaterialShader.VertexStage, self.VERTEX)
            self.setShaderSourceCode(QSGMaterialShader.FragmentStage, self.FRAG)
        else:
            from tempfile import NamedTemporaryFile

            with NamedTemporaryFile(delete=False, suffix=".vert") as vf:
                vf.write(self.VERTEX)
            with NamedTemporaryFile(delete=False, suffix=".frag") as ff:
                ff.write(self.FRAG)
            self.setShaderFileName(QSGMaterialShader.VertexStage, vf.name)
            self.setShaderFileName(QSGMaterialShader.FragmentStage, ff.name)
        self.setAttributeNames([b"aVertex", b"aOffset", b"aColor", b"aFlag"])

    def updateState(
        self, state, new_material, old_material
    ):  # pragma: no cover - Qt binding detail
        program = self.program()
        if new_material.offsets:
            program.enableAttributeArray(b"aOffset")
            program.setAttributeArray(b"aOffset", new_material.offsets)
        if new_material.colors:
            program.enableAttributeArray(b"aColor")
            program.setAttributeArray(b"aColor", new_material.colors)
        if new_material.flags:
            program.enableAttributeArray(b"aFlag")
            program.setAttributeArray(b"aFlag", new_material.flags)


class _EdgeMaterial(QSGMaterial):
    """Material supplying per-edge endpoints via instanced attributes."""

    _TYPE = QSGMaterialType()

    def __init__(self) -> None:
        super().__init__()
        self.color = QColor("gray")
        self.starts: List[QVector2D] = []
        self.ends: List[QVector2D] = []

    def type(self) -> QSGMaterialType:  # pragma: no cover - Qt binding detail
        return self._TYPE

    def createShader(
        self, render_mode: QSGRendererInterface.RenderMode
    ) -> QSGMaterialShader:  # pragma: no cover - Qt binding detail
        return _EdgeShader()


class _EdgeShader(QSGMaterialShader):
    """Render edges by interpolating per-instance start and end points."""

    VERTEX = QByteArray(
        b"""
        attribute highp vec4 aVertex;
        attribute highp vec2 aStart;
        attribute highp vec2 aEnd;
        varying highp vec2 vStart;
        varying highp vec2 vEnd;
        void main(){
            vStart = aStart;
            vEnd = aEnd;
            vec2 p = mix(aStart, aEnd, aVertex.x);
            gl_Position = vec4(p, 0.0, 1.0);
        }
        """
    )

    FRAG = QByteArray(
        b"""
        uniform lowp vec4 uColor;
        void main(){
            gl_FragColor = uColor;
        }
        """
    )

    def __init__(self) -> None:  # pragma: no cover - Qt binding detail
        super().__init__()
        if hasattr(self, "setShaderSourceCode"):
            self.setShaderSourceCode(QSGMaterialShader.VertexStage, self.VERTEX)
            self.setShaderSourceCode(QSGMaterialShader.FragmentStage, self.FRAG)
        else:
            from tempfile import NamedTemporaryFile

            with NamedTemporaryFile(delete=False, suffix=".vert") as vf:
                vf.write(self.VERTEX)
            with NamedTemporaryFile(delete=False, suffix=".frag") as ff:
                ff.write(self.FRAG)
            self.setShaderFileName(QSGMaterialShader.VertexStage, vf.name)
            self.setShaderFileName(QSGMaterialShader.FragmentStage, ff.name)
        self.setAttributeNames([b"aVertex", b"aStart", b"aEnd"])

    def updateState(
        self, state, new_material, old_material
    ):  # pragma: no cover - Qt binding detail
        program = self.program()
        program.setUniformValue("uColor", new_material.color)
        if new_material.starts:
            program.enableAttributeArray(b"aStart")
            program.setAttributeArray(b"aStart", new_material.starts)
        if new_material.ends:
            program.enableAttributeArray(b"aEnd")
            program.setAttributeArray(b"aEnd", new_material.ends)


@QmlElement
class GraphView(QQuickItem):
    """QQuickItem rendering nodes, edges and pulses via instanced attributes.

    Per-instance offset, color and flag data is supplied as geometry attributes
    so the scene no longer carries the 256-uniform limit. Large graphs render in
    a single draw call per primitive while level-of-detail behaviour such as
    antialiasing, label visibility and edge culling is governed by configurable
    zoom thresholds exposed as properties.
    """

    def __init__(self, parent: QQuickItem | None = None) -> None:
        super().__init__(parent)
        self.setFlag(QQuickItem.ItemHasContents, True)
        self._nodes: List[Tuple[float, float]] = []
        self._edges: List[Tuple[int, int]] = []
        self._node_offsets: List[QVector2D] = []
        self._node_geom: QSGGeometryNode | None = None
        self._node_material = _InstancedMaterial()
        self._edge_geom: QSGGeometryNode | None = None
        self._edge_material = _EdgeMaterial()
        self._pulse_geom: QSGGeometryNode | None = None
        self._pulse_material = _InstancedMaterial()
        self._pulses: Dict[int, int] = {}
        self._pulse_duration = 30
        self._node_colors: List[QColor] = []
        self._node_flags: List[float] = []
        self._node_labels: List[str] = []
        self._edges_dirty = True
        self._editable = True
        self._zoom = 1.0
        self._labels_visible = True
        self._edges_visible = True
        self._antialias_threshold = 0.5
        self._label_threshold = 0.3
        self._edge_threshold = 0.2
        self._update_lod()

    nodeModelChanged = Signal()

    def _get_node_model(self):
        """Return node data for QML bindings."""
        return [
            {"x": x, "y": y, "label": label}
            for (x, y), label in zip(self._nodes, self._node_labels)
        ]

    nodeModel = Property("QVariantList", _get_node_model, notify=nodeModelChanged)

    zoomChanged = Signal(float)
    labelsVisibleChanged = Signal(bool)
    edgesVisibleChanged = Signal(bool)

    def set_graph(
        self,
        nodes: Iterable[Tuple[float, float]],
        edges: Iterable[Tuple[int, int]],
        labels: Iterable[str] | None = None,
        colors: Iterable[str] | None = None,
        flags: Iterable[bool] | None = None,
    ) -> None:
        """Update geometry, instance attributes and schedule a redraw.

        Parameters
        ----------
        nodes:
            Iterable of ``(x, y)`` positions for each node.
        edges:
            Iterable of ``(a, b)`` node index pairs describing edges.
        labels:
            Optional iterable of text labels per node.
        colors:
            Optional iterable of colors recognised by :class:`QColor`.
        flags:
            Optional iterable of booleans marking node visibility.
        """

        self._nodes = list(nodes)
        self._edges = self._collapse_edges(edges)
        self._node_offsets = [QVector2D(x, y) for x, y in self._nodes]
        if colors is not None:
            self._node_colors = [QColor(c) for c in colors]
        else:
            self._node_colors = [QColor("white") for _ in self._nodes]
        if flags is not None:
            self._node_flags = [1.0 if f else 0.0 for f in flags]
        else:
            self._node_flags = [1.0 for _ in self._nodes]
        self._node_labels = list(labels) if labels else ["" for _ in self._nodes]
        self._edges_dirty = True
        rect = self._bounding_rect()
        try:
            self.update(rect)
        except TypeError:  # PySide binding without QRectF overload
            self.update()
        self.nodeModelChanged.emit()

    def apply_delta(self, delta: Dict[str, Dict[int, Tuple[float, float]]]) -> None:
        """Apply ``delta`` updates to buffers and schedule minimal repaint.

        Keys may include ``node_positions``, ``node_colors``, ``node_flags``,
        ``node_labels`` and ``edges``. Instance attribute buffers are updated
        in-place so only the latest values touch the GPU.
        """

        affected: Set[int] = set()

        positions = delta.get("node_positions", {})
        if positions:
            for idx, (x, y) in positions.items():
                i = int(idx)
                self._nodes[i] = (x, y)
                if i < len(self._node_offsets):
                    self._node_offsets[i].setX(x)
                    self._node_offsets[i].setY(y)
                affected.add(i)
            self._edges_dirty = True

        colors = delta.get("node_colors", {})
        for idx, color in colors.items():
            i = int(idx)
            self._node_colors[i] = QColor(color)
            affected.add(i)

        flags = delta.get("node_flags", {})
        for idx, flag in flags.items():
            i = int(idx)
            self._node_flags[i] = 1.0 if flag else 0.0
            affected.add(i)

        labels = delta.get("node_labels", {})
        for idx, text in labels.items():
            i = int(idx)
            self._node_labels[i] = str(text)
            affected.add(i)

        edges = delta.get("edges")
        if edges is not None:
            self._edges = self._collapse_edges(edges)
            self._edges_dirty = True
            for a, b in edges:
                affected.add(int(a))
                affected.add(int(b))

        closed = delta.get("closed_windows", [])
        for vid, _ in closed:
            i = int(vid)
            self._pulses[i] = self._pulse_duration
            affected.add(i)

        rect = self._bounding_rect(affected)
        try:
            self.update(rect)
        except TypeError:  # PySide binding without QRectF overload
            self.update()
        self.nodeModelChanged.emit()

    # --- level of detail -----------------------------------------------------
    def _update_lod(self) -> None:
        """Adjust antialiasing, label and edge visibility based on zoom."""
        aa = self._zoom > self._antialias_threshold
        if self.antialiasing() != aa:
            self.setAntialiasing(aa)
        old = self._labels_visible
        self._labels_visible = self._zoom > self._label_threshold
        if old != self._labels_visible:
            self.labelsVisibleChanged.emit(self._labels_visible)
        edge_old = self._edges_visible
        self._edges_visible = self._zoom > self._edge_threshold
        if edge_old != self._edges_visible:
            self.edgesVisibleChanged.emit(self._edges_visible)
            self._edges_dirty = True
            self.update()

    def _get_zoom(self) -> float:
        return self._zoom

    def _set_zoom(self, value: float) -> None:
        """Update zoom level, scale view and emit change signals."""
        self._zoom = value
        self.setScale(self._zoom)
        self._update_lod()
        self.update()
        self.zoomChanged.emit(self._zoom)

    zoom = Property(float, _get_zoom, _set_zoom, notify=zoomChanged)

    def _get_labels_visible(self) -> bool:
        return self._labels_visible

    labelsVisible = Property(bool, _get_labels_visible, notify=labelsVisibleChanged)

    def _get_edges_visible(self) -> bool:
        return self._edges_visible

    edgesVisible = Property(bool, _get_edges_visible, notify=edgesVisibleChanged)

    def _get_editable(self) -> bool:
        return self._editable

    def _set_editable(self, value: bool) -> None:
        """Enable or disable user interaction with the canvas."""
        self._editable = value

    editable = Property(bool, _get_editable, _set_editable)

    def _get_antialias_threshold(self) -> float:
        return self._antialias_threshold

    def _set_antialias_threshold(self, value: float) -> None:
        self._antialias_threshold = value
        self._update_lod()

    antialiasThreshold = Property(
        float, _get_antialias_threshold, _set_antialias_threshold
    )

    def _get_label_threshold(self) -> float:
        return self._label_threshold

    def _set_label_threshold(self, value: float) -> None:
        self._label_threshold = value
        self._update_lod()

    labelThreshold = Property(float, _get_label_threshold, _set_label_threshold)

    def _get_edge_threshold(self) -> float:
        return self._edge_threshold

    def _set_edge_threshold(self, value: float) -> None:
        self._edge_threshold = value
        self._update_lod()

    edgeThreshold = Property(float, _get_edge_threshold, _set_edge_threshold)

    # --- QQuickItem overrides -------------------------------------------------
    frameRendered = Signal()

    def wheelEvent(
        self, event: QWheelEvent
    ) -> None:  # pragma: no cover - Qt binding detail
        """Zoom the view in response to mouse wheel events."""
        factor = 1.0 + event.angleDelta().y() / 1200.0
        self.zoom = max(0.1, min(10.0, self._zoom * factor))
        event.accept()

    def updatePaintNode(
        self, old_node: QSGNode | None, data
    ) -> QSGNode:  # pragma: no cover - Qt binding detail
        root = old_node or QSGNode()
        self._update_edges(root)
        self._update_nodes(root)
        self._update_pulses(root)
        self.frameRendered.emit()
        return root

    # --- helpers --------------------------------------------------------------
    def _update_nodes(self, parent: QSGNode) -> None:
        """Render all nodes with a single instanced geometry node."""

        if self._node_geom is None:
            geom = QSGGeometryNode()
            geometry = QSGGeometry(QSGGeometry.defaultAttributes_Point2D(), 4)
            geometry.setDrawingMode(QSGGeometry.DrawTriangleStrip)
            verts = geometry.vertexDataAsPoint2D()
            verts[0].set(-0.5, -0.5)
            verts[1].set(0.5, -0.5)
            verts[2].set(-0.5, 0.5)
            verts[3].set(0.5, 0.5)
            geom.setGeometry(geometry)
            geom.setFlag(QSGNode.OwnsGeometry, True)
            geom.setMaterial(self._node_material)
            geom.setFlag(QSGNode.OwnsMaterial, True)
            parent.appendChildNode(geom)
            self._node_geom = geom

        self._node_material.offsets = self._node_offsets
        self._node_material.colors = [
            QVector4D(c.redF(), c.greenF(), c.blueF(), c.alphaF())
            for c in self._node_colors
        ]
        self._node_material.flags = self._node_flags
        if self._node_geom is not None and hasattr(self._node_geom, "setInstanceCount"):
            self._node_geom.setInstanceCount(len(self._node_offsets))

    def _update_pulses(self, parent: QSGNode) -> None:
        """Render and decay transient window-closure pulses."""

        offsets: List[QVector2D] = []
        colors: List[QVector4D] = []
        remove: List[int] = []
        for vid, ttl in self._pulses.items():
            if ttl <= 0:
                remove.append(vid)
                continue
            x, y = self._nodes[vid]
            offsets.append(QVector2D(x, y))
            alpha = ttl / float(self._pulse_duration)
            colors.append(QVector4D(1.0, 0.0, 0.0, alpha))
            self._pulses[vid] = ttl - 1
        for vid in remove:
            del self._pulses[vid]

        if self._pulse_geom is None:
            geom = QSGGeometryNode()
            geometry = QSGGeometry(QSGGeometry.defaultAttributes_Point2D(), 4)
            geometry.setDrawingMode(QSGGeometry.DrawTriangleStrip)
            verts = geometry.vertexDataAsPoint2D()
            verts[0].set(-0.6, -0.6)
            verts[1].set(0.6, -0.6)
            verts[2].set(-0.6, 0.6)
            verts[3].set(0.6, 0.6)
            geom.setGeometry(geometry)
            geom.setFlag(QSGNode.OwnsGeometry, True)
            geom.setMaterial(self._pulse_material)
            geom.setFlag(QSGNode.OwnsMaterial, True)
            parent.appendChildNode(geom)
            self._pulse_geom = geom

        self._pulse_material.offsets = offsets
        self._pulse_material.colors = colors
        self._pulse_material.flags = [1.0] * len(offsets)
        if self._pulse_geom is not None and hasattr(
            self._pulse_geom, "setInstanceCount"
        ):
            self._pulse_geom.setInstanceCount(len(offsets))

    def _update_edges(self, parent: QSGNode) -> None:
        """Render edges using per-instance start and end points when visible."""

        if not self._edges_visible or not self._edges:
            if self._edge_geom is not None and hasattr(
                self._edge_geom, "setInstanceCount"
            ):
                self._edge_geom.setInstanceCount(0)
            return

        if self._edge_geom is None:
            geom = QSGGeometryNode()
            geometry = QSGGeometry(QSGGeometry.defaultAttributes_Point2D(), 2)
            geometry.setDrawingMode(QSGGeometry.DrawLines)
            verts = geometry.vertexDataAsPoint2D()
            verts[0].set(0.0, 0.0)
            verts[1].set(1.0, 0.0)
            geom.setGeometry(geometry)
            geom.setFlag(QSGNode.OwnsGeometry, True)
            geom.setMaterial(self._edge_material)
            geom.setFlag(QSGNode.OwnsMaterial, True)
            parent.appendChildNode(geom)
            self._edge_geom = geom

        if not self._edges_dirty:
            return

        self._edge_material.starts = [
            QVector2D(*self._nodes[a]) for a, _ in self._edges
        ]
        self._edge_material.ends = [QVector2D(*self._nodes[b]) for _, b in self._edges]
        if self._edge_geom is not None and hasattr(self._edge_geom, "setInstanceCount"):
            self._edge_geom.setInstanceCount(len(self._edges))

        self._edges_dirty = False

    @Slot(str)
    @Slot(str, float, int)
    def save_snapshot(self, path: str, duration: float = 0.0, fps: int = 30) -> None:
        """Export the current view as an image or short video.

        Parameters
        ----------
        path:
            Destination file. The extension determines the format:
            ``.png`` for a static image or ``.mp4`` for a short clip.
        duration:
            Length of the MP4 clip in seconds. Must be positive for MP4 exports
            and is ignored for PNG exports.
        fps:
            Frame rate of the MP4 clip. Must be positive for MP4 exports and is
            ignored for PNG exports.
        """

        ext = path.lower().split(".")[-1]
        if ext not in {"png", "mp4"}:
            raise ValueError("Unsupported snapshot format; use .png or .mp4")
        if ext == "mp4":
            if duration <= 0:
                raise ValueError("MP4 snapshots require a positive duration")
            if fps <= 0:
                raise ValueError("MP4 snapshots require a positive fps")

        def _write(result) -> None:
            if ext == "png":
                result.saveToFile(path)
                return
            from PySide6.QtGui import QImage  # Local import to avoid GUI deps
            import numpy as np
            import imageio

            img = result.image().convertToFormat(QImage.Format_RGBA8888)
            width, height = img.width(), img.height()
            ptr = img.constBits()
            ptr.setsize(img.byteCount())
            arr = np.frombuffer(ptr, np.uint8).reshape(height, width, 4)
            frames = [arr] * max(1, int(duration * fps))
            imageio.mimsave(path, frames, fps=fps)

        self.grabToImage(_write)

    def _collapse_edges(
        self, edges: Iterable[Tuple[int, int]]
    ) -> List[Tuple[int, int]]:
        """Collapse parallel edges so only one instance per pair is kept."""

        seen: Set[Tuple[int, int]] = set()
        result: List[Tuple[int, int]] = []
        for a, b in edges:
            key = tuple(sorted((int(a), int(b))))
            if key in seen:
                continue
            seen.add(key)
            result.append((int(a), int(b)))
        return result

    def _bounding_rect(self, indices: Iterable[int] | None = None) -> QRectF:
        """Return bounding rectangle for ``indices`` or all nodes."""

        if not self._nodes:
            return QRectF()
        if indices:
            pts = [self._nodes[i] for i in indices if 0 <= i < len(self._nodes)]
            if not pts:
                return QRectF()
            xs, ys = zip(*pts)
        else:
            xs, ys = zip(*self._nodes)
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        return QRectF(xmin, ymin, xmax - xmin, ymax - ymin)
