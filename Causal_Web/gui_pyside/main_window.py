from __future__ import annotations

import os

from PySide6.QtCore import Qt, QTimer, QThread
from PySide6.QtGui import QAction, QKeySequence, QShortcut, QCloseEvent
from PySide6.QtWidgets import (
    QApplication,
    QDockWidget,
    QFileDialog,
    QFormLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSlider,
    QSpinBox,
    QLineEdit,
    QHBoxLayout,
    QWidget,
    QVBoxLayout,
)
import pyqtgraph as pg

from ..config import Config, EngineMode
from ..view import ViewSnapshot
from ..graph.io import load_graph, save_graph, new_graph
from ..graph.model import GraphModel
from ..gui.state import (
    get_active_file,
    get_graph,
    set_graph,
    set_active_file,
    clear_graph_dirty,
    is_graph_dirty,
    mark_graph_dirty,
)
from .canvas_widget import CanvasWidget
from .engine_profile_panel import EngineProfileDock
from .toolbar_builder import build_toolbar
from ..gui.command_stack import AddNodeCommand, AddObserverCommand
from ..engine.engine_v2.adapter import EngineAdapter
from .engine_worker import EngineWorker

tick_engine = EngineAdapter()


class GraphDockWidget(QDockWidget):
    """Dock widget that prompts about unsaved graph changes when closed."""

    def closeEvent(self, event: QCloseEvent) -> None:  # type: ignore[override]
        """Prompt to discard pending changes before closing."""
        from PySide6.QtWidgets import QMessageBox

        mw = self.parent()
        panels = []
        if hasattr(mw, "node_panel"):
            panels = [
                mw.node_panel,
                mw.connection_panel,
                mw.observer_panel,
                mw.meta_node_panel,
            ]
        dirty_panels = [p for p in panels if getattr(p, "dirty", False)]

        if is_graph_dirty():
            resp = QMessageBox.question(
                self,
                "Unapplied Changes",
                "There are unloaded changes to the Graph View. Close anyway?",
            )
            if resp != QMessageBox.Yes:
                event.ignore()
                return
        elif dirty_panels:
            resp = QMessageBox.question(
                self,
                "Unapplied Changes",
                "Discard changes to open panels?",
            )
            if resp != QMessageBox.Yes:
                event.ignore()
                return
        for p in panels:
            p.force_close()
        super().closeEvent(event)


class MainWindow(QMainWindow):
    """Main application window with dockable widgets.

    Attributes
    ----------
    graph_window : QMainWindow
        Hosts the toolbar and editing panels for the graph editor.
    """

    def __init__(self) -> None:
        """Initialize the main application window."""
        super().__init__()
        self.setWindowTitle("CWT Simulation Dashboard")
        self.resize(800, 600)

        self.sim_canvas = CanvasWidget(self, editable=False)
        self.setCentralWidget(self.sim_canvas)
        self.sim_canvas.load_model(get_graph())

        # graph editor dock, hidden by default
        self.canvas = CanvasWidget(self)

        # The graph editor window hosts dockable panels and the toolbar.
        # It must exist before building the toolbar as panel builders expect
        # ``self.graph_window`` to be available.
        self.graph_window = QMainWindow()

        self.adding_connection = False
        toolbar = build_toolbar(self)

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(toolbar)
        layout.addWidget(self.canvas)
        apply_btn = QPushButton("Apply Changes")
        apply_btn.clicked.connect(self._load_into_main)
        layout.addWidget(apply_btn)

        self.graph_window.setCentralWidget(central)

        self.canvas_dock = GraphDockWidget("Graph View", self)
        self.canvas_dock.setWidget(self.graph_window)
        self.canvas_dock.hide()
        self.addDockWidget(Qt.LeftDockWidgetArea, self.canvas_dock)
        self.canvas.load_model(GraphModel.from_dict(get_graph().to_dict()))

        self._undo_shortcut = QShortcut(QKeySequence("Ctrl+Z"), self)
        self._undo_shortcut.activated.connect(self.canvas.undo)
        self._redo_shortcut = QShortcut(QKeySequence("Ctrl+Y"), self)
        self._redo_shortcut.activated.connect(self.canvas.redo)

        self._create_menus()
        self.edit_action.setEnabled(bool(get_graph().nodes))
        self._create_docks()
        self._init_status_bar()
        self._init_engine_profile()
        self._init_telemetry()
        self._start_refresh_timer()

        self._engine_thread = QThread(self)
        self._engine_worker = EngineWorker(tick_engine)
        self._engine_worker.moveToThread(self._engine_thread)
        self._engine_thread.started.connect(self._engine_worker.run)
        self._engine_worker.snapshot_ready.connect(self._handle_engine_update)
        self._engine_thread.start()

    # ---- UI setup ----

    def closeEvent(self, event: QCloseEvent) -> None:  # type: ignore[override]
        """Stop the engine worker and thread before closing."""
        self._engine_worker.stop()
        self._engine_thread.quit()
        self._engine_thread.wait()
        super().closeEvent(event)

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

        edit_menu = menubar.addMenu("Edit")

        self.edit_action = QAction("Edit Graph...", self)
        self.edit_action.triggered.connect(self._show_graph_editor)
        self.edit_action.setEnabled(False)
        edit_menu.addAction(self.edit_action)

        undo_action = QAction("Undo", self)
        undo_action.triggered.connect(self.canvas.undo)
        edit_menu.addAction(undo_action)

        redo_action = QAction("Redo", self)
        redo_action.triggered.connect(self.canvas.redo)
        edit_menu.addAction(redo_action)

        settings_menu = menubar.addMenu("Settings")
        log_action = QAction("Log Files...", self)
        log_action.triggered.connect(self._show_log_files_window)
        settings_menu.addAction(log_action)

        analysis_menu = menubar.addMenu("Analysis")
        bell_action = QAction("Bell Inequality Analysis...", self)
        bell_action.triggered.connect(self._show_bell_analysis)
        analysis_menu.addAction(bell_action)

    def _create_docks(self) -> None:
        """Create and populate the control panel dock widgets."""
        dock = QDockWidget("Control Panel", self)
        dock.setFeatures(QDockWidget.NoDockWidgetFeatures)
        panel = QWidget()
        layout = QFormLayout(panel)

        self.tick_slider = QSlider(Qt.Horizontal)
        self.tick_slider.setMinimum(1)
        self.tick_slider.setMaximum(20)
        self.tick_slider.setValue(int(Config.tick_rate))
        self.tick_slider.valueChanged.connect(self._tick_rate_changed)

        self.tick_edit = QLineEdit(str(Config.tick_rate))
        self.tick_edit.editingFinished.connect(self._tick_rate_edited)

        tick_rate_row = QWidget()
        tick_rate_layout = QHBoxLayout(tick_rate_row)
        tick_rate_layout.setContentsMargins(0, 0, 0, 0)
        tick_rate_layout.addWidget(self.tick_slider)
        tick_rate_layout.addWidget(self.tick_edit)

        layout.addRow("Frame Rate", tick_rate_row)

        self.tick_label = QLabel("0")
        layout.addRow("Arrival Depth", self.tick_label)

        if Config.engine_mode == EngineMode.V2:
            self.depth_label = QLabel("0")
            layout.addRow("Depth", self.depth_label)
            self.window_label = QLabel("0")
            layout.addRow("Window", self.window_label)

        self.limit_spin = QSpinBox()
        self.limit_spin.setMinimum(1)
        self.limit_spin.setMaximum(100000)
        self.limit_spin.setValue(Config.max_ticks)
        layout.addRow("Depth Limit", self.limit_spin)

        self.start_button = QPushButton("Start Simulation")
        self.start_button.clicked.connect(self.start_simulation)
        self.start_button.setEnabled(get_active_file() is not None)
        layout.addRow(self.start_button)

        self.pause_button = QPushButton("Pause")
        self.pause_button.setEnabled(False)
        self.pause_button.clicked.connect(self.pause_or_resume)
        layout.addRow(self.pause_button)

        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_simulation)
        layout.addRow(self.stop_button)

        dock.setWidget(panel)
        self.addDockWidget(Qt.LeftDockWidgetArea, dock)

    def _init_status_bar(self) -> None:
        """Initialise the status bar with telemetry counters."""
        bar = self.statusBar()
        self._status_labels: dict[str, QLabel] = {
            "frame": QLabel("Frame: 0"),
            "events": QLabel("Events/sec: 0"),
            "gates": QLabel("Gates: 0"),
            "bridges": QLabel("Bridges: 0"),
            "residual": QLabel("Residual: 0"),
        }
        for lbl in self._status_labels.values():
            bar.addPermanentWidget(lbl)

    def _init_engine_profile(self) -> None:
        """Create dock showing the current engine configuration."""
        self.engine_profile = EngineProfileDock(self)
        self.addDockWidget(Qt.RightDockWidgetArea, self.engine_profile)

    def _init_telemetry(self) -> None:
        """Set up lightweight real-time plots for counters."""
        self.telemetry_dock = QDockWidget("Telemetry", self)
        self.telemetry_plot = pg.PlotWidget(title="Events/sec")
        self._telemetry_curve = self.telemetry_plot.plot(pen="y")
        self._telemetry_x: list[int] = []
        self._telemetry_y: list[float] = []
        self.telemetry_dock.setWidget(self.telemetry_plot)
        self.addDockWidget(Qt.RightDockWidgetArea, self.telemetry_dock)

    def _start_refresh_timer(self) -> None:
        """Begin periodic updates of the simulation canvas."""
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setInterval(100)
        self._refresh_timer.timeout.connect(self._refresh_sim_canvas)
        self._refresh_timer.start()

    def _handle_engine_update(self) -> None:
        """Update the UI when the worker emits a new snapshot.

        The simulation canvas reuses the live engine graph while running to
        avoid rebuilding a :class:`GraphModel` on every frame.
        """
        snap = self._engine_worker.latest_snapshot()
        if snap is None:
            return
        tick = tick_engine.current_frame()
        Config.current_tick = tick
        self.tick_label.setText(str(tick))
        if hasattr(self, "depth_label"):
            self.depth_label.setText(str(snap.frame))
        window = snap.counters.get("window", 0)
        if hasattr(self, "window_label"):
            self.window_label.setText(str(window))
        if hasattr(self.sim_canvas, "update_hud"):
            self.sim_canvas.update_hud(tick, snap.frame, window)
        self._update_status_bar(snap)
        self.engine_profile.update_snapshot()
        self._update_telemetry(snap)
        if snap.closed_windows and hasattr(self.sim_canvas, "highlight_closed_windows"):
            self.sim_canvas.highlight_closed_windows(snap.closed_windows)
        if Config.is_running:
            if self.sim_canvas.model is not tick_engine.graph:
                self.sim_canvas.model = tick_engine.graph
        else:
            self.sim_canvas.model = GraphModel.from_dict(get_graph().to_dict())
        self.sim_canvas.apply_diff(snap)

    def _refresh_sim_canvas(self) -> None:
        """Reload the canvas from the appropriate graph when idle."""
        with Config.state_lock:
            running = Config.is_running
            tick = Config.current_tick
        if Config.engine_mode == EngineMode.V2 and running:
            return
        model_dict = tick_engine.graph.to_dict() if running else get_graph().to_dict()
        self.tick_label.setText(str(tick))
        model = GraphModel.from_dict(model_dict)
        if Config.engine_mode == EngineMode.V2:
            snap = tick_engine.snapshot_for_ui()
            if hasattr(self, "depth_label"):
                self.depth_label.setText(str(snap.frame))
            window = snap.counters.get("window", 0)
            if hasattr(self, "window_label"):
                self.window_label.setText(str(window))
            if hasattr(self.sim_canvas, "update_hud"):
                self.sim_canvas.update_hud(tick, snap.frame, window)
            self._update_status_bar(snap)
            self.engine_profile.update_snapshot()
            self._update_telemetry(snap)
            if snap.closed_windows and hasattr(
                self.sim_canvas, "highlight_closed_windows"
            ):
                self.sim_canvas.highlight_closed_windows(snap.closed_windows)
            self.sim_canvas.model = model
            self.sim_canvas.apply_diff(snap)
        else:
            self.sim_canvas.load_model(model)
        if not running:
            self.start_button.setEnabled(get_active_file() is not None)
            self.pause_button.setEnabled(False)
            self.pause_button.setText("Pause")
            self.stop_button.setEnabled(False)

    # ---- actions ----

    def _tick_rate_changed(self, value: int) -> None:
        """Update ``Config.tick_rate`` from the slider."""
        Config.tick_rate = float(value)
        self.tick_edit.setText(str(float(value)))

    def _tick_rate_edited(self) -> None:
        """Synchronize slider with manual text input."""
        try:
            value = float(self.tick_edit.text())
        except ValueError:
            # reset invalid text
            self.tick_edit.setText(str(Config.tick_rate))
            return
        Config.tick_rate = value
        self.tick_slider.setValue(int(value))

    def _update_status_bar(self, snap: ViewSnapshot) -> None:
        """Refresh status bar telemetry from ``snap``."""
        self._status_labels["frame"].setText(f"Frame: {snap.frame}")
        self._status_labels["events"].setText(
            f"Events/sec: {snap.counters.get('events_per_sec', 0):.1f}"
        )
        self._status_labels["gates"].setText(
            f"Gates: {snap.counters.get('gates_fired', 0)}"
        )
        self._status_labels["bridges"].setText(
            f"Bridges: {snap.counters.get('active_bridges', 0)}"
        )
        self._status_labels["residual"].setText(
            f"Residual: {snap.counters.get('residual', 0):.2f}"
        )

    def _update_telemetry(self, snap: ViewSnapshot) -> None:
        """Append counters to real-time plots."""
        if "events_per_sec" in snap.counters:
            self._telemetry_x.append(snap.frame)
            self._telemetry_y.append(snap.counters["events_per_sec"])
            self._telemetry_x = self._telemetry_x[-3000:]
            self._telemetry_y = self._telemetry_y[-3000:]
            self._telemetry_curve.setData(self._telemetry_x, self._telemetry_y)

    def start_simulation(self) -> None:
        """Persist the active graph and launch the simulation thread.

        If the graph editor is open, any pending edits are applied to the
        simulation view before saving. This ensures the runtime graph matches
        the editor state when starting a new run.
        """
        # apply edits from the graph editor if it is currently visible
        if self.canvas_dock.isVisible():
            self._load_into_main()
        path = get_active_file() or Config.graph_file
        os.makedirs(os.path.dirname(path), exist_ok=True)
        save_graph(path, get_graph())
        # always write the runtime graph to the package input directory so the
        # engine uses the latest edits regardless of the loaded file location
        save_graph(Config.graph_file, get_graph())
        clear_graph_dirty()
        mark_graph_dirty()
        Config.new_run()
        Config.max_ticks = self.limit_spin.value()
        if Config.engine_mode == EngineMode.V2:
            tick_engine.build_graph(Config.graph_file)
            with Config.state_lock:
                if Config.is_running:
                    return
                Config.is_running = True
                Config.current_tick = 0
            tick_engine.start()
            self.start_button.setEnabled(False)
            self.pause_button.setEnabled(True)
            self.stop_button.setEnabled(True)
        else:
            tick_engine.build_graph()
            with Config.state_lock:
                if Config.is_running:
                    return
                Config.is_running = True
                tick = Config.current_tick
            tick_engine.simulation_loop()
            self.start_button.setEnabled(False)
            self.pause_button.setEnabled(True)
            self.stop_button.setEnabled(True)
            tick_engine._update_simulation_state(False, False, tick, None)

    def pause_or_resume(self) -> None:
        """Toggle between pausing and resuming the simulation."""
        with Config.state_lock:
            running = Config.is_running
        if Config.engine_mode == EngineMode.V2:
            if running:
                tick_engine.pause()
                with Config.state_lock:
                    Config.is_running = False
                self.pause_button.setText("Resume")
            else:
                tick_engine.start()
                with Config.state_lock:
                    Config.is_running = True
                self.pause_button.setText("Pause")
        else:
            if running:
                tick_engine.pause_simulation()
                self.pause_button.setText("Resume")
            else:
                tick_engine.resume_simulation()
                self.pause_button.setText("Pause")

    def stop_simulation(self) -> None:
        """Stop the simulation immediately."""
        if Config.engine_mode == EngineMode.V2:
            tick_engine.stop()
            with Config.state_lock:
                Config.is_running = False
        else:
            tick_engine.stop_simulation()
        self.start_button.setEnabled(get_active_file() is not None)
        self.pause_button.setEnabled(False)
        self.pause_button.setText("Pause")
        self.stop_button.setEnabled(False)

    def load_graph(self) -> None:
        """Load a graph from disk and display it."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Graph", Config.input_dir, "JSON Files (*.json)"
        )
        if not path:
            return
        try:
            graph = load_graph(path)
        except Exception as exc:
            from PySide6.QtWidgets import QMessageBox

            QMessageBox.warning(self, "Load Failed", str(exc))
            return
        set_graph(graph)
        set_active_file(path)
        clear_graph_dirty()
        self.start_button.setEnabled(True)
        self.sim_canvas.load_model(graph)
        # refresh editor if it is visible
        self.canvas.load_model(GraphModel.from_dict(graph.to_dict()))
        self.edit_action.setEnabled(True)
        self.start_button.setEnabled(True)

    def save_graph(self) -> None:
        """Write the current graph to disk."""
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Graph", Config.input_dir, "JSON Files (*.json)"
        )
        if not path:
            return
        try:
            save_graph(path, get_graph())
        except Exception as exc:
            from PySide6.QtWidgets import QMessageBox

            QMessageBox.warning(self, "Save Failed", str(exc))
            return
        set_active_file(path)
        clear_graph_dirty()

    def new_graph(self) -> None:
        """Create a new blank graph."""
        model = new_graph(True)
        set_graph(model)
        set_active_file(None)
        clear_graph_dirty()
        self.sim_canvas.load_model(model)
        self.canvas.load_model(GraphModel.from_dict(model.to_dict()))
        self.edit_action.setEnabled(True)
        self.start_button.setEnabled(False)

    def add_node(self) -> None:
        """Insert a new node and refresh the canvas."""
        model = get_graph()
        idx = 1
        while f"N{idx}" in model.nodes:
            idx += 1
        cmd = AddNodeCommand(model, f"N{idx}", {"x": 50.0 * idx, "y": 50.0 * idx})
        self.canvas.command_stack.do(cmd)
        self.canvas.load_model(model)

    def add_observer(self) -> None:
        """Create a new observer definition."""
        model = get_graph()
        idx = 1
        existing = {o.get("id") for o in model.observers}
        while f"OBS{idx}" in existing:
            idx += 1
        observer = {"id": f"OBS{idx}", "monitors": [], "frequency": 1.0}
        cmd = AddObserverCommand(model, observer)
        self.canvas.command_stack.do(cmd)
        self.canvas.load_model(model)
        self.observer_panel.open_new(cmd.index)

    def add_meta_node(self) -> None:
        """Create a new configured meta node."""
        model = get_graph()
        idx = 1
        while f"MN{idx}" in model.meta_nodes:
            idx += 1
        meta_id = f"MN{idx}"
        data = {
            "members": [],
            "constraints": {},
            "type": "Configured",
            "collapsed": False,
            "x": 0.0,
            "y": 0.0,
        }
        from ..gui.command_stack import AddMetaNodeCommand

        cmd = AddMetaNodeCommand(model, meta_id, data)
        self.canvas.command_stack.do(cmd)
        self.meta_node_panel.open_new(meta_id)

    def start_add_connection(self) -> None:
        """Toggle interactive connection mode for adding edges."""
        if self.adding_connection:
            self.canvas.cancel_connection_mode()
            if hasattr(self, "add_conn_action"):
                self.add_conn_action.setText("Add Connection")
            self.adding_connection = False
        else:
            self.canvas.enable_connection_mode()
            if hasattr(self, "add_conn_action"):
                self.add_conn_action.setText("Cancel Connection")
            self.adding_connection = True

    def finish_add_connection(self) -> None:
        """Reset the Add Connection button after a connection is created."""
        if hasattr(self, "add_conn_action"):
            self.add_conn_action.setText("Add Connection")
        self.adding_connection = False

    def _show_graph_editor(self) -> None:
        """Display the graph editor window."""
        self.canvas.load_model(GraphModel.from_dict(get_graph().to_dict()))
        self.canvas_dock.show()

    def _show_log_files_window(self) -> None:
        """Open the Log Files settings window."""
        if not hasattr(self, "log_files_window"):
            from .log_files_window import LogFilesWindow

            self.log_files_window = LogFilesWindow(self)
        self.log_files_window.show()

    def _show_bell_analysis(self) -> None:
        """Run Bell analysis and display the results."""
        if not hasattr(self, "bell_window"):
            from .bell_window import BellAnalysisWindow

            self.bell_window = BellAnalysisWindow(self)
        else:
            self.bell_window._run_analysis()  # refresh on repeat
        self.bell_window.show()

    def _load_into_main(self) -> None:
        """Apply the edited graph to the main simulation view and disk."""
        model = self.canvas.model
        set_graph(model)
        self.sim_canvas.load_model(model)
        path = get_active_file()
        if path:
            try:
                save_graph(path, model)
            except Exception as exc:
                from PySide6.QtWidgets import QMessageBox

                QMessageBox.warning(self, "Save Failed", str(exc))
            else:
                clear_graph_dirty()
        self.start_button.setEnabled(get_active_file() is not None)
        self.canvas_dock.hide()


def launch() -> None:
    """Entry point for the PySide6 GUI."""

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.show()
    app.exec()
