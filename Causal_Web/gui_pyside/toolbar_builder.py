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
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QToolBar,
    QWidget,
    QCheckBox,
    QLabel,
    QLineEdit,
    QVBoxLayout,
)

from ..gui.state import (
    get_graph,
    set_selected_node,
    set_selected_observer,
    mark_graph_dirty,
)

# ---------------------------------------------------------------------------
# Load tooltip text for GUI fields
_TOOLTIP_PATH = Path(__file__).resolve().parents[1] / "input" / "tooltip.json"
try:
    with open(_TOOLTIP_PATH, "r", encoding="utf-8") as fh:
        TOOLTIPS: Dict[str, str] = json.load(fh)
except FileNotFoundError:  # pragma: no cover - tooltips optional
    TOOLTIPS = {}

EVENT_TYPES = [
    "collapse",
    "law_wave",
    "emergence",
    "coherence",
    "phase",
    "region",
]


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

    def __init__(self, main_window, parent=None):
        super().__init__("Node", parent)
        self.main_window = main_window
        self.current: Optional[str] = None
        self.dirty = False
        self._force_close = False
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
            if field in {"x", "y"}:
                spin.setRange(-1_000_000, 1_000_000)
            label = TooltipLabel(field, TOOLTIPS.get(field))
            layout.addRow(label, spin)
            self.inputs[field] = spin
            spin.valueChanged.connect(self._mark_dirty)

        # tick source controls
        self.tick_source_cb = QCheckBox()
        layout.addRow(TooltipLabel("Tick Source"), self.tick_source_cb)
        self.tick_source_cb.toggled.connect(self._mark_dirty)

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
            spin.valueChanged.connect(self._mark_dirty)

        self.tick_source_cb.toggled.connect(self._toggle_tick_source_fields)

        apply_btn = QPushButton("Apply")
        apply_btn.clicked.connect(self.commit)
        layout.addRow(apply_btn)
        widget.installEventFilter(_FocusWatcher(self._minimize))
        self.setWidget(widget)

        # keep displayed coordinates in sync with the canvas
        self.main_window.canvas.node_position_changed.connect(self.update_position)

    def _minimize(self) -> None:
        """Hide the panel when it loses focus."""
        self.hide()

    def _mark_dirty(self, *args) -> None:
        """Indicate that edits are pending."""
        self.dirty = True

    def _toggle_tick_source_fields(self, checked: bool) -> None:
        for label, spin in self.ts_fields.values():
            label.setVisible(checked)
            spin.setVisible(checked)

    def show_node(self, node_id: str) -> None:
        if self.current == node_id:
            self.show()
            return

        if self.current and self.dirty:
            from PySide6.QtWidgets import QMessageBox

            resp = QMessageBox.question(
                self,
                "Unapplied Node Changes",
                "Apply changes to this node before switching?",
            )
            if resp == QMessageBox.Yes:
                self.commit()
            else:
                self.dirty = False

        model = get_graph()
        data = model.nodes.get(node_id)
        if data is None:
            return
        self.current = node_id
        self.dirty = False
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
        mark_graph_dirty()
        self.hide()
        self.current = None
        self.dirty = False

    def update_position(self, node_id: str, x: float, y: float) -> None:
        if self.current == node_id:
            self.inputs["x"].setValue(x)
            self.inputs["y"].setValue(y)

    def closeEvent(self, event) -> None:  # type: ignore[override]
        if self._force_close or not self.dirty:
            self._force_close = False
            return super().closeEvent(event)
        from PySide6.QtWidgets import QMessageBox

        resp = QMessageBox.question(
            self,
            "Unapplied Node Changes",
            "Discard changes to this node?",
        )
        if resp == QMessageBox.Yes:
            self.dirty = False
            super().closeEvent(event)
        else:
            event.ignore()

    def force_close(self) -> None:
        self._force_close = True
        self.close()


class ConnectionPanel(QDockWidget):
    """Dock widget for adding a connection between two nodes."""

    def __init__(self, main_window, parent=None):
        super().__init__("Connection", parent)
        self.main_window = main_window
        self.dirty = False
        self._force_close = False
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
        self.source_edit = QComboBox()
        self.target_edit = QComboBox()
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
            if hasattr(w, "valueChanged"):
                w.valueChanged.connect(self._mark_dirty)
            elif hasattr(w, "currentTextChanged"):
                w.currentTextChanged.connect(self._mark_dirty)

        # bridge widgets
        self.nodea_edit = QComboBox()
        self.nodeb_edit = QComboBox()
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
            if hasattr(w, "valueChanged"):
                w.valueChanged.connect(self._mark_dirty)
            elif hasattr(w, "currentTextChanged"):
                w.currentTextChanged.connect(self._mark_dirty)

        apply_btn = QPushButton("Apply")
        apply_btn.clicked.connect(self.commit)
        layout.addRow(apply_btn)
        widget.installEventFilter(_FocusWatcher(self._minimize))
        self.setWidget(widget)
        self.type_combo.currentIndexChanged.connect(self._update_fields)
        self.type_combo.currentIndexChanged.connect(self._mark_dirty)
        self._update_fields()

    def _mark_dirty(self, *args) -> None:
        self.dirty = True

    def _update_fields(self) -> None:
        is_edge = self.type_combo.currentText() == "Edge"
        for lbl, w in self.edge_widgets:
            lbl.setVisible(is_edge)
            w.setVisible(is_edge)
        for lbl, w in self.bridge_widgets:
            lbl.setVisible(not is_edge)
            w.setVisible(not is_edge)

    def _minimize(self) -> None:
        """Hide the connection panel when focus is lost."""
        self.hide()

    def _populate_node_lists(self) -> None:
        nodes = list(get_graph().nodes)
        for combo in (
            self.source_edit,
            self.target_edit,
            self.nodea_edit,
            self.nodeb_edit,
        ):
            combo.clear()
            combo.addItems(nodes)

    def open_for(self, source: str, target: str) -> None:
        if self.current_index is not None and self.dirty:
            from PySide6.QtWidgets import QMessageBox

            resp = QMessageBox.question(
                self,
                "Unapplied Connection Changes",
                "Apply changes to this connection before switching?",
            )
            if resp == QMessageBox.Yes:
                self.commit()
            else:
                self.dirty = False

        self._populate_node_lists()
        self.source = source
        self.target = target
        self.current_index = None
        self.current_type = "edge"
        self.type_combo.setCurrentIndex(0)
        self.source_edit.setCurrentText(source)
        self.target_edit.setCurrentText(target)
        self.atten_spin.setValue(1.0)
        self.density_spin.setValue(0.0)
        self.delay_spin.setValue(1.0)
        self.phase_shift_spin.setValue(0.0)
        self.weight_spin.setValue(0.0)
        self.dirty = False
        self.show()

    def show_connection(self, conn_type: str, index: int) -> None:
        """Display attributes for an existing connection."""
        if self.current_index == index and self.current_type == conn_type:
            self.show()
            return

        if self.current_index is not None and self.dirty:
            from PySide6.QtWidgets import QMessageBox

            resp = QMessageBox.question(
                self,
                "Unapplied Connection Changes",
                "Apply changes to this connection before switching?",
            )
            if resp == QMessageBox.Yes:
                self.commit()
            else:
                self.dirty = False

        model = get_graph()
        data = model.edges[index] if conn_type == "edge" else model.bridges[index]
        self.current_index = index
        self.current_type = conn_type
        self._populate_node_lists()
        self.type_combo.setCurrentIndex(0 if conn_type == "edge" else 1)
        if conn_type == "edge":
            self.source = data.get("from")
            self.target = data.get("to")
            self.source_edit.setCurrentText(self.source or "")
            self.target_edit.setCurrentText(self.target or "")
            self.atten_spin.setValue(float(data.get("attenuation", 1.0)))
            self.density_spin.setValue(float(data.get("density", 0.0)))
            self.delay_spin.setValue(float(data.get("delay", 1.0)))
            self.phase_shift_spin.setValue(float(data.get("phase_shift", 0.0)))
            self.weight_spin.setValue(float(data.get("weight", 0.0)))
        else:
            nodes = data.get("nodes", ["", ""])
            self.source, self.target = nodes[0], nodes[1]
            self.nodea_edit.setCurrentText(self.source)
            self.nodeb_edit.setCurrentText(self.target)
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
        self.dirty = False
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
                        self.source_edit.currentText(),
                        self.target_edit.currentText(),
                        delay=float(self.delay_spin.value()),
                        attenuation=float(self.atten_spin.value()),
                        density=float(self.density_spin.value()),
                        phase_shift=float(self.phase_shift_spin.value()),
                        weight=float(self.weight_spin.value()),
                        connection_type="edge",
                    )
                else:
                    model.add_connection(
                        self.nodea_edit.currentText(),
                        self.nodeb_edit.currentText(),
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
                            self.source_edit.currentText(),
                            self.target_edit.currentText(),
                            delay=float(self.delay_spin.value()),
                            attenuation=float(self.atten_spin.value()),
                            density=float(self.density_spin.value()),
                            phase_shift=float(self.phase_shift_spin.value()),
                            weight=float(self.weight_spin.value()),
                            connection_type="edge",
                        )
                    else:
                        model.add_connection(
                            self.nodea_edit.currentText(),
                            self.nodeb_edit.currentText(),
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
                                "from": self.source_edit.currentText(),
                                "to": self.target_edit.currentText(),
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
                            nodes=[
                                self.nodea_edit.currentText(),
                                self.nodeb_edit.currentText(),
                            ],
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
        mark_graph_dirty()
        self.hide()
        self.source = self.target = None
        self.current_index = None
        self.dirty = False

    def closeEvent(self, event) -> None:  # type: ignore[override]
        if self._force_close or not self.dirty:
            self._force_close = False
            return super().closeEvent(event)
        from PySide6.QtWidgets import QMessageBox

        resp = QMessageBox.question(
            self,
            "Unapplied Connection Changes",
            "Discard changes to this connection?",
        )
        if resp == QMessageBox.Yes:
            self.dirty = False
            super().closeEvent(event)
        else:
            event.ignore()

    def force_close(self) -> None:
        self._force_close = True
        self.close()


class ObserverPanel(QDockWidget):
    """Dock widget for editing observer definitions."""

    def __init__(self, main_window, parent=None):
        super().__init__("Observer", parent)
        self.main_window = main_window
        self.current_index: Optional[int] = None
        self.dirty = False
        self._force_close = False
        widget = QWidget()
        layout = QFormLayout(widget)

        self.id_edit = QLineEdit()
        layout.addRow(TooltipLabel("ID"), self.id_edit)
        self.id_edit.textChanged.connect(self._mark_dirty)

        self.monitor_checks: Dict[str, QCheckBox] = {}
        monitor_widget = QWidget()
        monitor_layout = QVBoxLayout(monitor_widget)
        for ev in EVENT_TYPES:
            cb = QCheckBox(ev)
            monitor_layout.addWidget(cb)
            self.monitor_checks[ev] = cb
            cb.toggled.connect(self._mark_dirty)
        layout.addRow(TooltipLabel("Monitors", "Event types to watch"), monitor_widget)

        self.freq_spin = QDoubleSpinBox()
        self.freq_spin.setDecimals(3)
        self.freq_spin.setValue(1.0)
        self.freq_spin.valueChanged.connect(self._mark_dirty)
        layout.addRow(
            TooltipLabel("Frequency", "How often (in ticks) the observer records data"),
            self.freq_spin,
        )

        self.node_list = QListWidget()
        self.node_list.setSelectionMode(QListWidget.MultiSelection)
        layout.addRow(
            TooltipLabel("Target Nodes", "A specific list of nodes to watch"),
            self.node_list,
        )
        self.node_list.itemSelectionChanged.connect(self._mark_dirty)

        apply_btn = QPushButton("Apply")
        apply_btn.clicked.connect(self.commit)
        layout.addRow(apply_btn)
        widget.installEventFilter(_FocusWatcher(self._minimize))
        self.setWidget(widget)

    def _minimize(self) -> None:
        """Hide the observer panel when focus is lost."""
        self.hide()

    def _mark_dirty(self, *args) -> None:
        """Mark the panel state as having unsaved changes."""
        self.dirty = True

    def open_for(self, index: int) -> None:
        if self.current_index == index:
            self.show()
            return

        if self.current_index is not None and self.dirty:
            from PySide6.QtWidgets import QMessageBox

            resp = QMessageBox.question(
                self,
                "Unapplied Observer Changes",
                "Apply changes to this observer before switching?",
            )
            if resp == QMessageBox.Yes:
                self.commit()
            else:
                self.dirty = False

        model = get_graph()
        if index < 0 or index >= len(model.observers):
            return
        self.current_index = index
        data = model.observers[index]
        self.id_edit.setText(data.get("id", ""))
        for ev, cb in self.monitor_checks.items():
            cb.setChecked(ev in data.get("monitors", []))
        self.freq_spin.setValue(float(data.get("frequency", 1.0)))
        self.node_list.clear()
        for nid in model.nodes:
            item = QListWidgetItem(nid)
            if nid in data.get("target_nodes", []):
                item.setSelected(True)
            self.node_list.addItem(item)
        self.dirty = False
        self.show()

    def open_new(self, index: int) -> None:
        if self.current_index is not None and self.dirty:
            from PySide6.QtWidgets import QMessageBox

            resp = QMessageBox.question(
                self,
                "Unapplied Observer Changes",
                "Apply changes to this observer before switching?",
            )
            if resp == QMessageBox.Yes:
                self.commit()
            else:
                self.dirty = False

        self.current_index = index
        self.id_edit.setText("")
        for cb in self.monitor_checks.values():
            cb.setChecked(False)
        self.freq_spin.setValue(1.0)
        self.node_list.clear()
        for nid in get_graph().nodes:
            self.node_list.addItem(QListWidgetItem(nid))
        self.dirty = False
        self.show()

    def commit(self) -> None:
        if self.current_index is None:
            return
        model = get_graph()
        existing = (
            model.observers[self.current_index]
            if self.current_index < len(model.observers)
            else {}
        )
        data = {
            "id": self.id_edit.text(),
            "monitors": [
                ev for ev, cb in self.monitor_checks.items() if cb.isChecked()
            ],
            "frequency": float(self.freq_spin.value()),
            "x": float(existing.get("x", 0.0)),
            "y": float(existing.get("y", 0.0)),
        }
        targets = [item.text() for item in self.node_list.selectedItems()]
        if targets:
            data["target_nodes"] = targets
        if self.current_index >= len(model.observers):
            model.observers.append(data)
        else:
            model.observers[self.current_index] = data
        self.main_window.canvas.load_model(model)
        set_selected_observer(self.current_index)
        mark_graph_dirty()
        self.hide()
        self.dirty = False

    def closeEvent(self, event) -> None:  # type: ignore[override]
        if self._force_close or not self.dirty:
            self._force_close = False
            return super().closeEvent(event)
        from PySide6.QtWidgets import QMessageBox

        resp = QMessageBox.question(
            self,
            "Unapplied Observer Changes",
            "Discard changes to this observer?",
        )
        if resp == QMessageBox.Yes:
            self.dirty = False
            super().closeEvent(event)
        else:
            event.ignore()

    def force_close(self) -> None:
        self._force_close = True
        self.close()


class MetaNodePanel(QDockWidget):
    """Dock widget for editing meta node definitions."""

    def __init__(self, main_window, parent=None):
        super().__init__("MetaNode", parent)
        self.main_window = main_window
        self.current: Optional[str] = None
        self.dirty = False
        self._force_close = False
        widget = QWidget()
        layout = QFormLayout(widget)

        self.id_label = QLabel()
        layout.addRow(TooltipLabel("ID"), self.id_label)

        self.member_list = QListWidget()
        self.member_list.setSelectionMode(QListWidget.MultiSelection)
        layout.addRow(
            TooltipLabel("Members", TOOLTIPS.get("members")), self.member_list
        )
        self.member_list.itemSelectionChanged.connect(self._mark_dirty)

        self.phase_tol = QDoubleSpinBox()
        self.phase_tol.setDecimals(3)
        layout.addRow(
            TooltipLabel("Tolerance", TOOLTIPS.get("tolerance")), self.phase_tol
        )
        self.phase_tol.valueChanged.connect(self._mark_dirty)

        self.min_coherence = QDoubleSpinBox()
        self.min_coherence.setDecimals(3)
        layout.addRow(
            TooltipLabel("Min Coherence", TOOLTIPS.get("min_coherence")),
            self.min_coherence,
        )
        self.min_coherence.valueChanged.connect(self._mark_dirty)

        self.shared_tick = QCheckBox()
        layout.addRow(
            TooltipLabel("Shared Tick Input", TOOLTIPS.get("shared_tick_input")),
            self.shared_tick,
        )
        self.shared_tick.toggled.connect(self._mark_dirty)

        self.sync_topology = QCheckBox()
        layout.addRow(
            TooltipLabel("Sync Topology", TOOLTIPS.get("sync_topology")),
            self.sync_topology,
        )
        self.sync_topology.toggled.connect(self._mark_dirty)

        self.role_lock = QLineEdit()
        layout.addRow(
            TooltipLabel("Role Lock", TOOLTIPS.get("role_lock")), self.role_lock
        )
        self.role_lock.textChanged.connect(self._mark_dirty)

        self.type_label = QLabel("Configured")
        layout.addRow(TooltipLabel("Type", TOOLTIPS.get("type")), self.type_label)

        self.collapsed_check = QCheckBox()
        layout.addRow(
            TooltipLabel("Collapsed", TOOLTIPS.get("collapsed")), self.collapsed_check
        )
        self.collapsed_check.toggled.connect(self._mark_dirty)

        apply_btn = QPushButton("Apply")
        apply_btn.clicked.connect(self.commit)
        layout.addRow(apply_btn)
        widget.installEventFilter(_FocusWatcher(self._minimize))
        self.setWidget(widget)

    def _minimize(self) -> None:
        """Hide the meta node panel when focus is lost."""
        self.hide()

    def _mark_dirty(self, *args) -> None:
        """Indicate that changes need saving."""
        self.dirty = True

    def open_new(self, meta_id: str) -> None:
        if self.current and self.dirty:
            from PySide6.QtWidgets import QMessageBox

            resp = QMessageBox.question(
                self,
                "Unapplied MetaNode Changes",
                "Apply changes to this meta node before switching?",
            )
            if resp == QMessageBox.Yes:
                self.commit()
            else:
                self.dirty = False

        self.current = meta_id
        self.id_label.setText(meta_id)
        self.phase_tol.setValue(0.0)
        self.min_coherence.setValue(0.0)
        self.shared_tick.setChecked(False)
        self.sync_topology.setChecked(False)
        self.role_lock.setText("")
        self.collapsed_check.setChecked(False)
        self.member_list.clear()
        for nid in get_graph().nodes:
            self.member_list.addItem(QListWidgetItem(nid))
        self.dirty = False
        self.show()

    def show_meta_node(self, meta_id: str) -> None:
        if self.current == meta_id:
            self.show()
            return

        if self.current and self.dirty:
            from PySide6.QtWidgets import QMessageBox

            resp = QMessageBox.question(
                self,
                "Unapplied MetaNode Changes",
                "Apply changes to this meta node before switching?",
            )
            if resp == QMessageBox.Yes:
                self.commit()
            else:
                self.dirty = False

        model = get_graph()
        data = model.meta_nodes.get(meta_id)
        if data is None:
            return
        self.current = meta_id
        self.id_label.setText(meta_id)
        self.member_list.clear()
        for nid in model.nodes:
            item = QListWidgetItem(nid)
            if nid in data.get("members", []):
                item.setSelected(True)
            self.member_list.addItem(item)
        cons = data.get("constraints", {})
        self.phase_tol.setValue(float(cons.get("phase_lock", {}).get("tolerance", 0.0)))
        self.min_coherence.setValue(
            float(cons.get("coherence_tie", {}).get("min_coherence", 0.0))
        )
        self.shared_tick.setChecked(bool(cons.get("shared_tick_input")))
        self.sync_topology.setChecked(bool(cons.get("sync_topology")))
        rl = cons.get("role_lock")
        if isinstance(rl, list):
            self.role_lock.setText(",".join(str(r) for r in rl))
        elif rl is not None:
            self.role_lock.setText(str(rl))
        else:
            self.role_lock.setText("")
        self.collapsed_check.setChecked(bool(data.get("collapsed", False)))
        self.dirty = False
        self.show()

    def commit(self) -> None:
        if not self.current:
            return
        model = get_graph()
        meta = model.meta_nodes.get(self.current, {})
        meta["members"] = [item.text() for item in self.member_list.selectedItems()]
        constraints: Dict[str, Any] = {}
        tol = float(self.phase_tol.value())
        if tol:
            constraints["phase_lock"] = {"tolerance": tol}
        coh = float(self.min_coherence.value())
        if coh:
            constraints["coherence_tie"] = {"min_coherence": coh}
        if self.shared_tick.isChecked():
            constraints["shared_tick_input"] = True
        if self.sync_topology.isChecked():
            constraints["sync_topology"] = True
        rl_text = self.role_lock.text().strip()
        if rl_text:
            if rl_text.lower() in {"true", "false"}:
                constraints["role_lock"] = rl_text.lower() == "true"
            else:
                constraints["role_lock"] = [
                    s.strip() for s in rl_text.split(",") if s.strip()
                ]
        meta["constraints"] = constraints
        meta["type"] = "Configured"
        meta["collapsed"] = self.collapsed_check.isChecked()
        if "x" not in meta:
            meta["x"] = 0.0
        if "y" not in meta:
            meta["y"] = 0.0
        model.meta_nodes[self.current] = meta
        self.main_window.canvas.load_model(model)
        mark_graph_dirty()
        self.hide()
        self.dirty = False

    def closeEvent(self, event) -> None:  # type: ignore[override]
        if self._force_close or not self.dirty:
            self._force_close = False
            return super().closeEvent(event)
        from PySide6.QtWidgets import QMessageBox

        resp = QMessageBox.question(
            self,
            "Unapplied MetaNode Changes",
            "Discard changes to this meta node?",
        )
        if resp == QMessageBox.Yes:
            self.dirty = False
            super().closeEvent(event)
        else:
            event.ignore()

    def force_close(self) -> None:
        self._force_close = True
        self.close()


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

    add_obs_action = QAction("Add Observer", main_window)
    add_obs_action.triggered.connect(main_window.add_observer)
    toolbar.addAction(add_obs_action)

    add_meta_action = QAction("Add MetaNode", main_window)
    add_meta_action.triggered.connect(main_window.add_meta_node)
    toolbar.addAction(add_meta_action)

    layout_action = QAction("Auto Layout", main_window)
    layout_action.triggered.connect(main_window.canvas.auto_layout)
    toolbar.addAction(layout_action)

    main_window.node_panel = NodePanel(main_window, main_window.graph_window)
    main_window.graph_window.addDockWidget(
        Qt.RightDockWidgetArea, main_window.node_panel
    )
    main_window.node_panel.hide()

    main_window.connection_panel = ConnectionPanel(
        main_window, main_window.graph_window
    )
    main_window.graph_window.addDockWidget(
        Qt.RightDockWidgetArea, main_window.connection_panel
    )
    main_window.connection_panel.hide()

    main_window.observer_panel = ObserverPanel(main_window, main_window.graph_window)
    main_window.graph_window.addDockWidget(
        Qt.RightDockWidgetArea, main_window.observer_panel
    )
    main_window.observer_panel.hide()

    main_window.meta_node_panel = MetaNodePanel(main_window, main_window.graph_window)
    main_window.graph_window.addDockWidget(
        Qt.RightDockWidgetArea, main_window.meta_node_panel
    )
    main_window.meta_node_panel.hide()

    main_window.canvas.node_selected.connect(main_window.node_panel.show_node)
    main_window.canvas.connection_request.connect(main_window.connection_panel.open_for)
    main_window.canvas.connection_selected.connect(
        main_window.connection_panel.show_connection
    )
    main_window.canvas.meta_node_selected.connect(
        main_window.meta_node_panel.show_meta_node
    )
    main_window.canvas.observer_selected.connect(main_window.observer_panel.open_for)

    return toolbar
