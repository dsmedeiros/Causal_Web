from __future__ import annotations

import os

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QDockWidget,
    QFileDialog,
    QFormLayout,
    QMainWindow,
    QPushButton,
    QSlider,
    QToolBar,
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
from .canvas_widget import CanvasWidget
from ..engine import tick_engine


class MainWindow(QMainWindow):
    """Main application window with dockable widgets."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("CWT Simulation Dashboard")
        self.resize(800, 600)

        self.setCentralWidget(QWidget())
        self.canvas = CanvasWidget(self)
        self.canvas_dock = QDockWidget("Graph View", self)
        self.canvas_dock.setWidget(self.canvas)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.canvas_dock)
        self.canvas.load_model(get_graph())

        self._undo_shortcut = QShortcut(QKeySequence("Ctrl+Z"), self)
        self._undo_shortcut.activated.connect(self.canvas.undo)
        self._redo_shortcut = QShortcut(QKeySequence("Ctrl+Y"), self)
        self._redo_shortcut.activated.connect(self.canvas.redo)

        self._create_menus()
        self._create_toolbars()
        self._create_docks()

    # ---- UI setup ----

    def _create_menus(self) -> None:
        """Create the File and Edit menus for graph actions."""

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

        edit_menu = menubar.addMenu("Edit")

        layout_action = QAction("Auto Layout", self)
        layout_action.triggered.connect(self.canvas.auto_layout)
        edit_menu.addAction(layout_action)

        undo_action = QAction("Undo", self)
        undo_action.triggered.connect(self.canvas.undo)
        edit_menu.addAction(undo_action)

        redo_action = QAction("Redo", self)
        redo_action.triggered.connect(self.canvas.redo)
        edit_menu.addAction(redo_action)

    def _create_toolbars(self) -> None:
        """Create the toolbar with the Connect action."""

        toolbar = QToolBar("Graph", self)
        self.addToolBar(toolbar)

        connect_action = QAction("Connect", self)
        connect_action.triggered.connect(self.canvas.enable_connection_mode)
        toolbar.addAction(connect_action)

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
