"""Utility for constructing toolbars and property panels for the Qt GUI."""

from __future__ import annotations

from typing import Optional, Any, Dict

from pathlib import Path
import json

from PySide6.QtCore import QObject, Qt, QEvent, QTimer
from PySide6.QtGui import QAction, QCursor
from PySide6.QtWidgets import (
    QComboBox,
    QDockWidget,
    QDoubleSpinBox,
    QFormLayout,
    QPushButton,
    QToolBar,
    QWidget,
    QCheckBox,
    QLabel,
    QLineEdit,
)

from ..gui.state import get_graph, set_selected_node

# ---------------------------------------------------------------------------
# Load tooltip text for GUI fields
_TOOLTIP_PATH = Path(__file__).resolve().parents[1] / "input" / "tooltip.json"
try:
    with open(_TOOLTIP_PATH, "r", encoding="utf-8") as fh:
        TOOLTIPS: Dict[str, str] = json.load(fh)
except FileNotFoundError:  # pragma: no cover - tooltips optional
    TOOLTIPS = {}


class TooltipLabel(QLabel):
    """QLabel displaying a tooltip after 0.5s of hover."""

    def __init__(self, text: str, tip: str | None = None) -> None:
        super().__init__(text)
        self.tip = tip or ""
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._show_tooltip)

    def enterEvent(self, event) -> None:  # type: ignore[override]
        if self.tip:
            self._timer.start(500)
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:  # type: ignore[override]
        self._timer.stop()
        QLabel.setToolTip(self, "")
        super().leaveEvent(event)

    def _show_tooltip(self) -> None:
        if self.tip:
            from PySide6.QtWidgets import QToolTip

            pos = self.mapToGlobal(self.rect().bottomRight())
            QToolTip.showText(pos, self.tip, self)


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

        self.inputs: Dict[str, QDoubleSpinBox] = {}
        for field in [
            "x",
            "y",
            "frequency",
            "refractory_period",
            "base_threshold",
            "phase",
        ]:
            spin = QDoubleSpinBox()
            spin.setDecimals(3)
            label = TooltipLabel(field, TOOLTIPS.get(field))
            layout.addRow(label, spin)
            self.inputs[field] = spin

        # tick source controls
        self.tick_source_cb = QCheckBox()
        layout.addRow(TooltipLabel("Tick Source"), self.tick_source_cb)

        self.ts_fields: Dict[str, tuple[TooltipLabel, QDoubleSpinBox]] = {}
        for field, label_text in [
            ("tick_interval", "Tick Interval"),
            ("tick_phase", "Phase"),
            ("end_tick", "End Tick"),
        ]:
            spin = QDoubleSpinBox()
            spin.setDecimals(3)
            label = TooltipLabel(label_text, TOOLTIPS.get(field))
            layout.addRow(label, spin)
            label.hide()
            spin.hide()
            self.ts_fields[field] = (label, spin)

        self.tick_source_cb.toggled.connect(self._toggle_tick_source_fields)

        apply_btn = QPushButton("Apply")
        apply_btn.clicked.connect(self.commit)
        layout.addRow(apply_btn)
        widget.installEventFilter(_FocusWatcher(self.commit))
        self.setWidget(widget)
        # update coordinates if node moves while panel is visible
        self.main_window.canvas.node_position_changed.connect(self.update_position)

    def _toggle_tick_source_fields(self, checked: bool) -> None:
        for label, spin in self.ts_fields.values():
            label.setVisible(checked)
            spin.setVisible(checked)

    def show_node(self, node_id: str) -> None:
        model = get_graph()
        data = model.nodes.get(node_id)
        if data is None:
            return
        self.current = node_id
        for key, spin in self.inputs.items():
            spin.setValue(float(data.get(key, 0.0)))

        # tick source info
        ts_rec = next(
            (s for s in model.tick_sources if s.get("node_id") == node_id), None
        )
        self.tick_source_cb.setChecked(ts_rec is not None)
        self._toggle_tick_source_fields(ts_rec is not None)
        if ts_rec:
            self.ts_fields["tick_interval"][1].setValue(
                float(ts_rec.get("tick_interval", 1.0))
            )
            self.ts_fields["tick_phase"][1].setValue(float(ts_rec.get("phase", 0.0)))
            self.ts_fields["end_tick"][1].setValue(float(ts_rec.get("end_tick", 0.0)))
        else:
            for _, spin in self.ts_fields.values():
                spin.setValue(0.0)

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

        # update tick source record
        ts_rec = next(
            (s for s in model.tick_sources if s.get("node_id") == self.current), None
        )
        if self.tick_source_cb.isChecked():
            data = {
                "node_id": self.current,
                "tick_interval": float(self.ts_fields["tick_interval"][1].value()),
                "phase": float(self.ts_fields["tick_phase"][1].value()),
                "end_tick": float(self.ts_fields["end_tick"][1].value()),
            }
            if ts_rec:
                ts_rec.update(data)
            else:
                model.tick_sources.append(data)
        elif ts_rec:
            model.tick_sources.remove(ts_rec)

        self.main_window.canvas.load_model(model)
        set_selected_node(self.current)
        self.hide()
        self.current = None

    def update_position(self, node_id: str, x: float, y: float) -> None:
        if self.current == node_id:
            self.inputs["x"].setValue(x)
            self.inputs["y"].setValue(y)


class ConnectionPanel(QDockWidget):
    """Dock widget for adding a connection between two nodes."""

    def __init__(self, main_window):
        super().__init__("Connection", main_window)
        self.main_window = main_window
        self.source: Optional[str] = None
        self.target: Optional[str] = None
        self.current_index: Optional[int] = None
        self.current_type: str = "edge"
        widget = QWidget()
        layout = QFormLayout(widget)
        self.type_combo = QComboBox()
        self.type_combo.addItems(["Edge", "Bridge"])
        layout.addRow("Type", self.type_combo)

        # edge widgets
        self.source_edit = QLineEdit()
        self.target_edit = QLineEdit()
        self.atten_spin = QDoubleSpinBox()
        self.atten_spin.setValue(1.0)
        self.density_spin = QDoubleSpinBox()
        self.density_spin.setValue(0.0)
        self.delay_spin = QDoubleSpinBox()
        self.delay_spin.setValue(1.0)
        self.phase_shift_spin = QDoubleSpinBox()
        self.phase_shift_spin.setDecimals(3)
        self.weight_spin = QDoubleSpinBox()
        self.weight_spin.setDecimals(3)

        self.edge_widgets = [
            (TooltipLabel("Source ID"), self.source_edit),
            (TooltipLabel("Target ID"), self.target_edit),
            (TooltipLabel("Attenuation", TOOLTIPS.get("attenuation")), self.atten_spin),
            (TooltipLabel("Density", TOOLTIPS.get("density")), self.density_spin),
            (TooltipLabel("Delay", TOOLTIPS.get("delay")), self.delay_spin),
            (
                TooltipLabel("Phase Shift", TOOLTIPS.get("phase_shift")),
                self.phase_shift_spin,
            ),
            (TooltipLabel("Weight", TOOLTIPS.get("weight")), self.weight_spin),
        ]
        for lbl, w in self.edge_widgets:
            layout.addRow(lbl, w)

        # bridge widgets
        self.nodea_edit = QLineEdit()
        self.nodeb_edit = QLineEdit()
        self.bridge_type_combo = QComboBox()
        self.bridge_type_combo.addItems(["Braided"])
        self.phase_offset_spin = QDoubleSpinBox()
        self.drift_tol_spin = QDoubleSpinBox()
        self.decoherence_spin = QDoubleSpinBox()
        self.initial_strength_spin = QDoubleSpinBox()
        self.medium_type_edit = QLineEdit()
        self.mutable_check = QCheckBox()

        self.bridge_widgets = [
            (TooltipLabel("Node A ID"), self.nodea_edit),
            (TooltipLabel("Node B ID"), self.nodeb_edit),
            (TooltipLabel("Bridge Type"), self.bridge_type_combo),
            (
                TooltipLabel("Phase Offset", TOOLTIPS.get("phase_offset")),
                self.phase_offset_spin,
            ),
            (
                TooltipLabel("Drift Tolerance", TOOLTIPS.get("drift_tolerance")),
                self.drift_tol_spin,
            ),
            (
                TooltipLabel("Decoherence Limit", TOOLTIPS.get("decoherence_limit")),
                self.decoherence_spin,
            ),
            (
                TooltipLabel("Initial Strength", TOOLTIPS.get("initial_strength")),
                self.initial_strength_spin,
            ),
            (
                TooltipLabel("Medium Type", TOOLTIPS.get("medium_type")),
                self.medium_type_edit,
            ),
            (TooltipLabel("Mutable", TOOLTIPS.get("mutable")), self.mutable_check),
        ]
        for lbl, w in self.bridge_widgets:
            layout.addRow(lbl, w)

        apply_btn = QPushButton("Apply")
        apply_btn.clicked.connect(self.commit)
        layout.addRow(apply_btn)
        widget.installEventFilter(_FocusWatcher(self.commit))
        self.setWidget(widget)
        self.type_combo.currentIndexChanged.connect(self._update_fields)
        self._update_fields()

    def _update_fields(self) -> None:
        is_edge = self.type_combo.currentText() == "Edge"
        for lbl, w in self.edge_widgets:
            lbl.setVisible(is_edge)
            w.setVisible(is_edge)
        for lbl, w in self.bridge_widgets:
            lbl.setVisible(not is_edge)
            w.setVisible(not is_edge)

    def open_for(self, source: str, target: str) -> None:
        self.source = source
        self.target = target
        self.current_index = None
        self.current_type = "edge"
        self.type_combo.setCurrentIndex(0)
        self.source_edit.setText(source)
        self.target_edit.setText(target)
        self.atten_spin.setValue(1.0)
        self.density_spin.setValue(0.0)
        self.delay_spin.setValue(1.0)
        self.phase_shift_spin.setValue(0.0)
        self.weight_spin.setValue(0.0)
        self.show()

    def show_connection(self, conn_type: str, index: int) -> None:
        """Display attributes for an existing connection."""

        model = get_graph()
        data = model.edges[index] if conn_type == "edge" else model.bridges[index]
        self.current_index = index
        self.current_type = conn_type
        self.type_combo.setCurrentIndex(0 if conn_type == "edge" else 1)
        if conn_type == "edge":
            self.source = data.get("from")
            self.target = data.get("to")
            self.source_edit.setText(self.source or "")
            self.target_edit.setText(self.target or "")
            self.atten_spin.setValue(float(data.get("attenuation", 1.0)))
            self.density_spin.setValue(float(data.get("density", 0.0)))
            self.delay_spin.setValue(float(data.get("delay", 1.0)))
            self.phase_shift_spin.setValue(float(data.get("phase_shift", 0.0)))
            self.weight_spin.setValue(float(data.get("weight", 0.0)))
        else:
            nodes = data.get("nodes", ["", ""])
            self.source, self.target = nodes[0], nodes[1]
            self.nodea_edit.setText(self.source)
            self.nodeb_edit.setText(self.target)
            self.bridge_type_combo.setCurrentText(data.get("bridge_type", "Braided"))
            self.phase_offset_spin.setValue(float(data.get("phase_offset", 0.0)))
            self.drift_tol_spin.setValue(float(data.get("drift_tolerance", 0.0)))
            self.decoherence_spin.setValue(float(data.get("decoherence_limit", 0.0)))
            self.initial_strength_spin.setValue(
                float(data.get("initial_strength", 0.0))
            )
            self.medium_type_edit.setText(data.get("medium_type", ""))
            self.mutable_check.setChecked(bool(data.get("mutable", False)))
        self._update_fields()
        self.show()

    def commit(self) -> None:
        if not self.source or not self.target:
            return
        model = get_graph()
        conn_type = "edge" if self.type_combo.currentText() == "Edge" else "bridge"
        try:
            if self.current_index is None:
                if conn_type == "edge":
                    model.add_connection(
                        self.source_edit.text(),
                        self.target_edit.text(),
                        delay=float(self.delay_spin.value()),
                        attenuation=float(self.atten_spin.value()),
                        density=float(self.density_spin.value()),
                        phase_shift=float(self.phase_shift_spin.value()),
                        weight=float(self.weight_spin.value()),
                        connection_type="edge",
                    )
                else:
                    model.add_connection(
                        self.nodea_edit.text(),
                        self.nodeb_edit.text(),
                        connection_type="bridge",
                        bridge_type=self.bridge_type_combo.currentText(),
                        phase_offset=float(self.phase_offset_spin.value()),
                        drift_tolerance=float(self.drift_tol_spin.value()),
                        decoherence_limit=float(self.decoherence_spin.value()),
                        initial_strength=float(self.initial_strength_spin.value()),
                        medium_type=self.medium_type_edit.text(),
                        mutable=self.mutable_check.isChecked(),
                    )
            else:
                if conn_type != self.current_type:
                    model.remove_connection(self.current_index, self.current_type)
                    self.current_index = None
                    if conn_type == "edge":
                        model.add_connection(
                            self.source_edit.text(),
                            self.target_edit.text(),
                            delay=float(self.delay_spin.value()),
                            attenuation=float(self.atten_spin.value()),
                            density=float(self.density_spin.value()),
                            phase_shift=float(self.phase_shift_spin.value()),
                            weight=float(self.weight_spin.value()),
                            connection_type="edge",
                        )
                    else:
                        model.add_connection(
                            self.nodea_edit.text(),
                            self.nodeb_edit.text(),
                            connection_type="bridge",
                            bridge_type=self.bridge_type_combo.currentText(),
                            phase_offset=float(self.phase_offset_spin.value()),
                            drift_tolerance=float(self.drift_tol_spin.value()),
                            decoherence_limit=float(self.decoherence_spin.value()),
                            initial_strength=float(self.initial_strength_spin.value()),
                            medium_type=self.medium_type_edit.text(),
                            mutable=self.mutable_check.isChecked(),
                        )
                else:
                    if conn_type == "edge":
                        model.update_connection(
                            self.current_index,
                            "edge",
                            **{
                                "from": self.source_edit.text(),
                                "to": self.target_edit.text(),
                                "delay": float(self.delay_spin.value()),
                                "attenuation": float(self.atten_spin.value()),
                                "density": float(self.density_spin.value()),
                                "phase_shift": float(self.phase_shift_spin.value()),
                                "weight": float(self.weight_spin.value()),
                            },
                        )
                    else:
                        model.update_connection(
                            self.current_index,
                            "bridge",
                            nodes=[self.nodea_edit.text(), self.nodeb_edit.text()],
                            bridge_type=self.bridge_type_combo.currentText(),
                            phase_offset=float(self.phase_offset_spin.value()),
                            drift_tolerance=float(self.drift_tol_spin.value()),
                            decoherence_limit=float(self.decoherence_spin.value()),
                            initial_strength=float(self.initial_strength_spin.value()),
                            medium_type=self.medium_type_edit.text(),
                            mutable=self.mutable_check.isChecked(),
                        )
        except Exception as exc:  # pragma: no cover - GUI feedback
            print(f"Failed to add connection: {exc}")
        else:
            self.main_window.canvas.load_model(model)
        self.hide()
        self.source = self.target = None
        self.current_index = None


def build_toolbar(main_window) -> QToolBar:
    """Create the graph editing toolbar and property docks.

    The returned :class:`QToolBar` contains actions for adding nodes,
    creating connections and applying automatic layout.  The caller is
    responsible for embedding the toolbar, typically inside the Graph View
    dock.  File and edit menu actions are handled separately by the main
    window.
    """

    toolbar = QToolBar("Graph", main_window)

    add_node_action = QAction("Add Node", main_window)
    add_node_action.triggered.connect(main_window.add_node)
    toolbar.addAction(add_node_action)

    add_conn_action = QAction("Add Connection", main_window)
    add_conn_action.triggered.connect(main_window.start_add_connection)
    toolbar.addAction(add_conn_action)

    layout_action = QAction("Auto Layout", main_window)
    layout_action.triggered.connect(main_window.canvas.auto_layout)
    toolbar.addAction(layout_action)

    main_window.node_panel = NodePanel(main_window)
    main_window.addDockWidget(Qt.RightDockWidgetArea, main_window.node_panel)
    main_window.node_panel.hide()

    main_window.connection_panel = ConnectionPanel(main_window)
    main_window.addDockWidget(Qt.RightDockWidgetArea, main_window.connection_panel)
    main_window.connection_panel.hide()

    main_window.canvas.node_selected.connect(main_window.node_panel.show_node)
    main_window.canvas.connection_request.connect(main_window.connection_panel.open_for)
    main_window.canvas.connection_selected.connect(
        main_window.connection_panel.show_connection
    )

    return toolbar
