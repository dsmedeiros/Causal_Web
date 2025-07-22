from __future__ import annotations

from dataclasses import dataclass
from typing import Dict
import os

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QBrush, QPen, QAction
from PySide6.QtWidgets import (
    QApplication,
    QDockWidget,
    QFileDialog,
    QFormLayout,
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsLineItem,
    QGraphicsScene,
    QGraphicsView,
    QMainWindow,
    QPushButton,
    QSlider,
    QWidget,
)

from ..config import Config
from ..graph.io import load_graph, save_graph, new_graph
from ..graph.model import GraphModel
from ..gui.state import (
    get_active_file,
    get_graph,
    set_graph,
    set_active_file,
)
from ..engine import tick_engine


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
        self.edges: list[EdgeItem] = []

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionChange:
            for edge in self.edges:
                edge.update_position()
        return super().itemChange(change, value)


class EdgeItem(QGraphicsLineItem):
    """Line connecting two NodeItems."""

    def __init__(self, source: NodeItem, target: NodeItem):
        super().__init__()
        self.source = source
        self.target = target
        pen = QPen(Qt.darkGray)
        pen.setWidth(2)
        self.setPen(pen)
        self.update_position()
        source.edges.append(self)
        target.edges.append(self)

    def update_position(self):
        self.setLine(self.source.x(), self.source.y(), self.target.x(), self.target.y())


@dataclass
class GraphCanvas:
    """Manage a QGraphicsScene for a GraphModel."""

    scene: QGraphicsScene
    nodes: Dict[str, NodeItem]

    def __init__(self, scene: QGraphicsScene):
        self.scene = scene
        self.nodes = {}

    def load_model(self, model: GraphModel) -> None:
        self.scene.clear()
        self.nodes.clear()
        for node_id, data in model.nodes.items():
            x, y = data.get("x", 0.0), data.get("y", 0.0)
            item = NodeItem(node_id, x, y)
            self.scene.addItem(item)
            self.nodes[node_id] = item
        for edge in model.edges:
            src = self.nodes.get(edge.get("from"))
            dst = self.nodes.get(edge.get("to"))
            if src and dst:
                self.scene.addItem(EdgeItem(src, dst))


class MainWindow(QMainWindow):
    """Main application window with dockable widgets."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("CWT Simulation Dashboard")
        self.resize(800, 600)

        self.scene = QGraphicsScene(self)
        self.view = QGraphicsView(self.scene)
        self.setCentralWidget(self.view)
        self.canvas = GraphCanvas(self.scene)
        self.canvas.load_model(get_graph())

        self._create_menus()
        self._create_docks()

    # ---- UI setup ----

    def _create_menus(self) -> None:
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")

        load_action = QAction("Load", self)
        load_action.triggered.connect(self.load_graph)
        file_menu.addAction(load_action)

        save_action = QAction("Save", self)
        save_action.triggered.connect(self.save_graph)
        file_menu.addAction(save_action)

        new_action = QAction("New", self)
        new_action.triggered.connect(self.new_graph)
        file_menu.addAction(new_action)

    def _create_docks(self) -> None:
        dock = QDockWidget("Control Panel", self)
        panel = QWidget()
        layout = QFormLayout(panel)

        self.tick_slider = QSlider(Qt.Horizontal)
        self.tick_slider.setMinimum(1)
        self.tick_slider.setMaximum(20)
        self.tick_slider.setValue(int(Config.tick_rate))
        self.tick_slider.valueChanged.connect(self._tick_rate_changed)
        layout.addRow("Tick Rate", self.tick_slider)

        self.start_button = QPushButton("Start Simulation")
        self.start_button.clicked.connect(self.start_simulation)
        layout.addRow(self.start_button)

        dock.setWidget(panel)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)

    # ---- actions ----

    def _tick_rate_changed(self, value: int) -> None:
        """Update ``Config.tick_rate`` from the slider."""
        Config.tick_rate = float(value)

    def start_simulation(self) -> None:
        """Save the graph and start the simulation loop."""
        path = get_active_file() or Config.input_path("graph.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        save_graph(path, get_graph())
        Config.new_run()
        tick_engine.build_graph()
        with Config.state_lock:
            if Config.is_running:
                return
            Config.is_running = True
            tick = Config.current_tick
        tick_engine.simulation_loop()
        self.start_button.setEnabled(False)
        tick_engine._update_simulation_state(False, False, tick, None)

    def load_graph(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Graph", Config.input_dir, "JSON Files (*.json)"
        )
        if not path:
            return
        try:
            graph = load_graph(path)
        except Exception as exc:
            print(f"Failed to load graph: {exc}")
            return
        set_graph(graph)
        set_active_file(path)
        self.canvas.load_model(graph)

    def save_graph(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Graph", Config.input_dir, "JSON Files (*.json)"
        )
        if not path:
            return
        try:
            save_graph(path, get_graph())
        except Exception as exc:
            print(f"Failed to save graph: {exc}")
            return
        set_active_file(path)

    def new_graph(self):
        model = new_graph(True)
        set_graph(model)
        set_active_file(None)
        self.canvas.load_model(model)


def launch() -> None:
    """Entry point for the PySide6 GUI."""

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.show()
    app.exec()
