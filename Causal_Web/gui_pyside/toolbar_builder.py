"""Utility for constructing toolbars and property panels for the Qt GUI."""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QObject, Qt, QEvent
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QComboBox,
    QDockWidget,
    QDoubleSpinBox,
    QFormLayout,
    QToolBar,
    QWidget,
)

from ..gui.state import get_graph, set_selected_node


class _FocusWatcher(QObject):
    """Call a function when the watched widget loses focus."""

    def __init__(self, callback):
        super().__init__()
        self.callback = callback

    def eventFilter(self, obj, event):  # type: ignore[override]
        if event.type() == QEvent.FocusOut:
            self.callback()
        return False


class NodePanel(QDockWidget):
    """Dock widget for editing node attributes."""

    def __init__(self, main_window):
        super().__init__("Node", main_window)
        self.main_window = main_window
        self.current: Optional[str] = None
        widget = QWidget()
        layout = QFormLayout(widget)
        self.inputs = {}
        for field in ["x", "y", "frequency", "refractory_period", "base_threshold"]:
            spin = QDoubleSpinBox()
            spin.setDecimals(3)
            layout.addRow(field, spin)
            self.inputs[field] = spin
        widget.installEventFilter(_FocusWatcher(self.commit))
        self.setWidget(widget)

    def show_node(self, node_id: str) -> None:
        model = get_graph()
        data = model.nodes.get(node_id)
        if data is None:
            return
        self.current = node_id
        for key, spin in self.inputs.items():
            spin.setValue(float(data.get(key, 0.0)))
        self.show()

    def commit(self) -> None:
        if not self.current:
            return
        model = get_graph()
        node = model.nodes.get(self.current)
        if node is None:
            return
        for key, spin in self.inputs.items():
            node[key] = float(spin.value())
        self.main_window.canvas.load_model(model)
        set_selected_node(self.current)
        self.hide()
        self.current = None


class ConnectionPanel(QDockWidget):
    """Dock widget for adding a connection between two nodes."""

    def __init__(self, main_window):
        super().__init__("Connection", main_window)
        self.main_window = main_window
        self.source: Optional[str] = None
        self.target: Optional[str] = None
        widget = QWidget()
        layout = QFormLayout(widget)
        self.type_combo = QComboBox()
        self.type_combo.addItems(["Edge", "Bridge"])
        self.delay_spin = QDoubleSpinBox()
        self.delay_spin.setValue(1.0)
        self.atten_spin = QDoubleSpinBox()
        self.atten_spin.setValue(1.0)
        layout.addRow("Type", self.type_combo)
        layout.addRow("Delay", self.delay_spin)
        layout.addRow("Attenuation", self.atten_spin)
        widget.installEventFilter(_FocusWatcher(self.commit))
        self.setWidget(widget)

    def open_for(self, source: str, target: str) -> None:
        self.source = source
        self.target = target
        self.type_combo.setCurrentIndex(0)
        self.delay_spin.setValue(1.0)
        self.atten_spin.setValue(1.0)
        self.show()

    def commit(self) -> None:
        if not self.source or not self.target:
            return
        model = get_graph()
        try:
            model.add_connection(
                self.source,
                self.target,
                delay=float(self.delay_spin.value()),
                attenuation=float(self.atten_spin.value()),
                connection_type=(
                    "edge" if self.type_combo.currentText() == "Edge" else "bridge"
                ),
            )
        except Exception as exc:  # pragma: no cover - GUI feedback
            print(f"Failed to add connection: {exc}")
        else:
            self.main_window.canvas.load_model(model)
        self.hide()
        self.source = self.target = None


def build_toolbar(main_window) -> QToolBar:
    """Create the graph toolbar and attach property panels.

    The toolbar now only exposes graph editing tools such as adding nodes or
    connections. File and edit actions are provided by the menus on the main
    window.
    """

    toolbar = QToolBar("Graph", main_window)
    main_window.addToolBar(toolbar)

    add_node_action = QAction("Add Node", main_window)
    add_node_action.triggered.connect(main_window.add_node)
    toolbar.addAction(add_node_action)

    add_conn_action = QAction("Add Connection", main_window)
    add_conn_action.triggered.connect(main_window.start_add_connection)
    toolbar.addAction(add_conn_action)

    main_window.node_panel = NodePanel(main_window)
    main_window.addDockWidget(Qt.RightDockWidgetArea, main_window.node_panel)
    main_window.node_panel.hide()

    main_window.connection_panel = ConnectionPanel(main_window)
    main_window.addDockWidget(Qt.RightDockWidgetArea, main_window.connection_panel)
    main_window.connection_panel.hide()

    main_window.canvas.node_selected.connect(main_window.node_panel.show_node)
    main_window.canvas.connection_request.connect(main_window.connection_panel.open_for)

    return toolbar
