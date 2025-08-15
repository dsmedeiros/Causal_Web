"""Service objects for GUI panel logic."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from PySide6.QtWidgets import (
    QDockWidget,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QLineEdit,
    QCheckBox,
    QPushButton,
    QWidget,
    QLabel,
    QListWidget,
)

from .shared import TooltipLabel, TOOLTIPS, _FocusWatcher
from ..gui.state import get_graph, mark_graph_dirty


@dataclass
class NodePanelSetupService:
    """Build the ``NodePanel`` widgets."""

    panel: QDockWidget
    main_window: Any

    def build(self) -> None:
        self._init_state()
        widget = QWidget()
        layout = QFormLayout(widget)
        self._build_node_fields(layout)
        self._build_tick_source(layout)
        self._finish(widget)
        self.panel.main_window.canvas.node_position_changed.connect(
            self.panel.update_position
        )

    # ------------------------------------------------------------------
    def _init_state(self) -> None:
        p = self.panel
        p.main_window = self.main_window
        p.current = None
        p.dirty = False
        p._force_close = False

    # ------------------------------------------------------------------
    def _build_node_fields(self, layout) -> None:
        self.panel.inputs = {}
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
            layout.addRow(TooltipLabel(field, TOOLTIPS.get(field)), spin)
            self.panel.inputs[field] = spin
            spin.valueChanged.connect(self.panel._mark_dirty)

        self.panel.self_connect_cb = QCheckBox()
        layout.addRow(TooltipLabel("Self Connect"), self.panel.self_connect_cb)
        self.panel.self_connect_cb.toggled.connect(self.panel._mark_dirty)

    # ------------------------------------------------------------------
    def _build_tick_source(self, layout) -> None:
        p = self.panel
        p.tick_source_cb = QCheckBox()
        layout.addRow(TooltipLabel("Frame Source"), p.tick_source_cb)
        p.tick_source_cb.toggled.connect(p._mark_dirty)

        p.ts_fields = {}
        for field, label_text in [
            ("tick_interval", "Frame Interval"),
            ("tick_phase", "Phase"),
            ("end_tick", "End Frame"),
        ]:
            spin = QDoubleSpinBox()
            spin.setDecimals(3)
            label = TooltipLabel(label_text, TOOLTIPS.get(field))
            layout.addRow(label, spin)
            label.hide()
            spin.hide()
            p.ts_fields[field] = (label, spin)
            spin.valueChanged.connect(p._mark_dirty)

        p.tick_source_cb.toggled.connect(p._toggle_tick_source_fields)

    # ------------------------------------------------------------------
    def _finish(self, widget) -> None:
        apply_btn = QPushButton("Apply")
        apply_btn.clicked.connect(self.panel.commit)
        layout = widget.layout()
        layout.addRow(apply_btn)
        widget.installEventFilter(_FocusWatcher(self.panel._minimize))
        self.panel.setWidget(widget)


@dataclass
class ConnectionPanelSetupService:
    """Build the ``ConnectionPanel`` widgets."""

    panel: QDockWidget
    main_window: Any
    parent: Any = None

    def build(self) -> None:
        self._init_state()
        widget = QWidget()
        layout = QFormLayout(widget)
        self._build_type_selector(layout)
        self._build_edge_widgets(layout)
        self._build_bridge_widgets(layout)
        self._finish(widget, layout)

    # ------------------------------------------------------------------
    def _init_state(self) -> None:
        p = self.panel
        p.main_window = self.main_window
        p.dirty = False
        p._force_close = False
        p.source = None
        p.target = None
        p.current_index = None
        p.current_type = "edge"

    # ------------------------------------------------------------------
    def _build_type_selector(self, layout) -> None:
        from PySide6.QtWidgets import QComboBox

        self.panel.type_combo = QComboBox()
        self.panel.type_combo.addItems(["Edge", "Bridge"])
        layout.addRow("Type", self.panel.type_combo)

    # ------------------------------------------------------------------
    def _build_edge_widgets(self, layout) -> None:
        p = self.panel
        p.source_edit = QComboBox()
        p.target_edit = QComboBox()
        p.atten_spin = QDoubleSpinBox()
        p.atten_spin.setValue(1.0)
        p.density_spin = QDoubleSpinBox()
        p.density_spin.setValue(0.0)
        p.delay_spin = QDoubleSpinBox()
        p.delay_spin.setValue(1.0)
        p.phase_shift_spin = QDoubleSpinBox()
        p.phase_shift_spin.setDecimals(3)
        p.weight_spin = QDoubleSpinBox()
        p.weight_spin.setDecimals(3)

        p.edge_widgets = [
            (TooltipLabel("Source ID"), p.source_edit),
            (TooltipLabel("Target ID"), p.target_edit),
            (
                TooltipLabel("Attenuation", TOOLTIPS.get("attenuation")),
                p.atten_spin,
            ),
            (TooltipLabel("Density", TOOLTIPS.get("density")), p.density_spin),
            (TooltipLabel("Delay", TOOLTIPS.get("delay")), p.delay_spin),
            (
                TooltipLabel("Phase Shift", TOOLTIPS.get("phase_shift")),
                p.phase_shift_spin,
            ),
            (TooltipLabel("Weight", TOOLTIPS.get("weight")), p.weight_spin),
        ]
        for lbl, w in p.edge_widgets:
            layout.addRow(lbl, w)
            if hasattr(w, "valueChanged"):
                w.valueChanged.connect(p._mark_dirty)
            elif hasattr(w, "currentTextChanged"):
                w.currentTextChanged.connect(p._mark_dirty)

    # ------------------------------------------------------------------
    def _build_bridge_widgets(self, layout) -> None:
        p = self.panel
        p.nodea_edit = QComboBox()
        p.nodeb_edit = QComboBox()
        p.bridge_type_combo = QComboBox()
        p.bridge_type_combo.addItems(["Braided"])
        p.phase_offset_spin = QDoubleSpinBox()
        p.drift_tol_spin = QDoubleSpinBox()
        p.decoherence_spin = QDoubleSpinBox()
        p.initial_strength_spin = QDoubleSpinBox()
        p.medium_type_edit = QLineEdit()
        p.mutable_check = QCheckBox()
        p.entangled_check = QCheckBox()
        p.entangled_id_label = QLineEdit()
        p.entangled_id_label.setReadOnly(True)

        p.bridge_widgets = [
            (TooltipLabel("Node A ID"), p.nodea_edit),
            (TooltipLabel("Node B ID"), p.nodeb_edit),
            (TooltipLabel("Bridge Type"), p.bridge_type_combo),
            (
                TooltipLabel("Phase Offset", TOOLTIPS.get("phase_offset")),
                p.phase_offset_spin,
            ),
            (
                TooltipLabel("Drift Tolerance", TOOLTIPS.get("drift_tolerance")),
                p.drift_tol_spin,
            ),
            (
                TooltipLabel("Decoherence Limit", TOOLTIPS.get("decoherence_limit")),
                p.decoherence_spin,
            ),
            (
                TooltipLabel("Initial Strength", TOOLTIPS.get("initial_strength")),
                p.initial_strength_spin,
            ),
            (
                TooltipLabel("Medium Type", TOOLTIPS.get("medium_type")),
                p.medium_type_edit,
            ),
            (
                TooltipLabel("Mutable", TOOLTIPS.get("mutable")),
                p.mutable_check,
            ),
            (
                TooltipLabel("Entanglement Enabled"),
                p.entangled_check,
            ),
            (
                TooltipLabel("Entangled ID"),
                p.entangled_id_label,
            ),
        ]
        for lbl, w in p.bridge_widgets:
            layout.addRow(lbl, w)
            if hasattr(w, "valueChanged"):
                w.valueChanged.connect(p._mark_dirty)
            elif hasattr(w, "currentTextChanged"):
                w.currentTextChanged.connect(p._mark_dirty)

    # ------------------------------------------------------------------
    def _finish(self, widget, layout) -> None:
        apply_btn = QPushButton("Apply")
        apply_btn.clicked.connect(self.panel.commit)
        layout.addRow(apply_btn)
        widget.installEventFilter(_FocusWatcher(self.panel._minimize))
        self.panel.setWidget(widget)
        tc = self.panel.type_combo
        tc.currentIndexChanged.connect(self.panel._update_fields)
        tc.currentIndexChanged.connect(self.panel._mark_dirty)
        self.panel._update_fields()


@dataclass
class MetaNodePanelSetupService:
    """Build widgets for :class:`~Causal_Web.gui_legacy.toolbar_builder.MetaNodePanel`."""

    panel: QDockWidget
    main_window: Any

    def build(self) -> None:
        self._init_state()
        widget = QWidget()
        layout = QFormLayout(widget)
        self._build_member_fields(layout)
        self._build_constraint_fields(layout)
        self._finish(widget, layout)

    # ------------------------------------------------------------------
    def _init_state(self) -> None:
        p = self.panel
        p.main_window = self.main_window
        p.current = None
        p.dirty = False
        p._force_close = False

    # ------------------------------------------------------------------
    def _build_member_fields(self, layout) -> None:
        p = self.panel
        p.id_label = QLabel()
        layout.addRow(TooltipLabel("ID"), p.id_label)

        p.member_list = QListWidget()
        p.member_list.setSelectionMode(QListWidget.MultiSelection)
        layout.addRow(TooltipLabel("Members", TOOLTIPS.get("members")), p.member_list)
        p.member_list.itemSelectionChanged.connect(p._mark_dirty)

    # ------------------------------------------------------------------
    def _build_constraint_fields(self, layout) -> None:
        p = self.panel
        p.phase_tol = QDoubleSpinBox()
        p.phase_tol.setDecimals(3)
        layout.addRow(TooltipLabel("Tolerance", TOOLTIPS.get("tolerance")), p.phase_tol)
        p.phase_tol.valueChanged.connect(p._mark_dirty)

        p.min_coherence = QDoubleSpinBox()
        p.min_coherence.setDecimals(3)
        layout.addRow(
            TooltipLabel("Min Coherence", TOOLTIPS.get("min_coherence")),
            p.min_coherence,
        )
        p.min_coherence.valueChanged.connect(p._mark_dirty)

        p.shared_tick = QCheckBox()
        layout.addRow(
            TooltipLabel("Shared Frame Input", TOOLTIPS.get("shared_tick_input")),
            p.shared_tick,
        )
        p.shared_tick.toggled.connect(p._mark_dirty)

        p.sync_topology = QCheckBox()
        layout.addRow(
            TooltipLabel("Sync Topology", TOOLTIPS.get("sync_topology")),
            p.sync_topology,
        )
        p.sync_topology.toggled.connect(p._mark_dirty)

        p.role_lock = QLineEdit()
        layout.addRow(TooltipLabel("Role Lock", TOOLTIPS.get("role_lock")), p.role_lock)
        p.role_lock.textChanged.connect(p._mark_dirty)

        p.type_label = QLabel("Configured")
        layout.addRow(TooltipLabel("Type", TOOLTIPS.get("type")), p.type_label)

        p.collapsed_check = QCheckBox()
        layout.addRow(
            TooltipLabel("Collapsed", TOOLTIPS.get("collapsed")),
            p.collapsed_check,
        )
        p.collapsed_check.toggled.connect(p._mark_dirty)

    # ------------------------------------------------------------------
    def _finish(self, widget, layout) -> None:
        apply_btn = QPushButton("Apply")
        apply_btn.clicked.connect(self.panel.commit)
        layout.addRow(apply_btn)
        widget.installEventFilter(_FocusWatcher(self.panel._minimize))
        self.panel.setWidget(widget)


@dataclass
class ConnectionCommitService:
    """Handle connection creation and updates."""

    panel: Any
    model: Any

    def commit(self) -> Any:
        conn_type = (
            "edge" if self.panel.type_combo.currentText() == "Edge" else "bridge"
        )
        try:
            if self.panel.current_index is None:
                self._add(conn_type)
            elif conn_type != self.panel.current_type:
                self._replace(conn_type)
            else:
                self._update(conn_type)
        except Exception as exc:  # pragma: no cover - GUI feedback
            print(f"Failed to add connection: {exc}")
        else:
            self.panel.main_window.canvas.load_model(self.model)
        mark_graph_dirty()
        return self.model

    # ------------------------------------------------------------------
    def _add(self, conn_type: str) -> None:
        if conn_type == "edge":
            self.model.add_connection(
                self.panel.source_edit.currentText(),
                self.panel.target_edit.currentText(),
                delay=float(self.panel.delay_spin.value()),
                attenuation=float(self.panel.atten_spin.value()),
                density=float(self.panel.density_spin.value()),
                phase_shift=float(self.panel.phase_shift_spin.value()),
                weight=float(self.panel.weight_spin.value()),
                connection_type="edge",
            )
        else:
            self.model.add_connection(
                self.panel.nodea_edit.currentText(),
                self.panel.nodeb_edit.currentText(),
                connection_type="bridge",
                bridge_type=self.panel.bridge_type_combo.currentText(),
                phase_offset=float(self.panel.phase_offset_spin.value()),
                drift_tolerance=float(self.panel.drift_tol_spin.value()),
                decoherence_limit=float(self.panel.decoherence_spin.value()),
                initial_strength=float(self.panel.initial_strength_spin.value()),
                medium_type=self.panel.medium_type_edit.text(),
                mutable=self.panel.mutable_check.isChecked(),
                is_entangled=self.panel.entangled_check.isChecked(),
                entangled_id=self.panel.entangled_id_label.text() or None,
            )

    # ------------------------------------------------------------------
    def _replace(self, conn_type: str) -> None:
        self.model.remove_connection(self.panel.current_index, self.panel.current_type)
        self.panel.current_index = None
        self._add(conn_type)

    # ------------------------------------------------------------------
    def _update(self, conn_type: str) -> None:
        if conn_type == "edge":
            self.model.update_connection(
                self.panel.current_index,
                "edge",
                **{
                    "from": self.panel.source_edit.currentText(),
                    "to": self.panel.target_edit.currentText(),
                    "delay": float(self.panel.delay_spin.value()),
                    "attenuation": float(self.panel.atten_spin.value()),
                    "density": float(self.panel.density_spin.value()),
                    "phase_shift": float(self.panel.phase_shift_spin.value()),
                    "weight": float(self.panel.weight_spin.value()),
                },
            )
        else:
            self.model.update_connection(
                self.panel.current_index,
                "bridge",
                nodes=[
                    self.panel.nodea_edit.currentText(),
                    self.panel.nodeb_edit.currentText(),
                ],
                bridge_type=self.panel.bridge_type_combo.currentText(),
                phase_offset=float(self.panel.phase_offset_spin.value()),
                drift_tolerance=float(self.panel.drift_tol_spin.value()),
                decoherence_limit=float(self.panel.decoherence_spin.value()),
                initial_strength=float(self.panel.initial_strength_spin.value()),
                medium_type=self.panel.medium_type_edit.text(),
                mutable=self.panel.mutable_check.isChecked(),
                is_entangled=self.panel.entangled_check.isChecked(),
                entangled_id=self.panel.entangled_id_label.text() or None,
            )


@dataclass
class ConnectionDisplayService:
    """Populate the connection panel with existing data."""

    panel: Any
    conn_type: str
    index: int

    def display(self) -> None:
        self._maybe_handle_dirty()
        data = self._fetch()
        self._fill_fields(data)
        self._finish()

    # ------------------------------------------------------------------
    def _maybe_handle_dirty(self) -> None:
        if self.panel.current_index is not None and self.panel.dirty:
            from PySide6.QtWidgets import QMessageBox

            resp = QMessageBox.question(
                self.panel,
                "Unapplied Connection Changes",
                "Apply changes to this connection before switching?",
            )
            if resp == QMessageBox.Yes:
                self.panel.commit()
            else:
                self.panel.dirty = False

    # ------------------------------------------------------------------
    def _fetch(self):
        model = get_graph()
        if self.conn_type == "edge":
            return model.edges[self.index]
        return model.bridges[self.index]

    # ------------------------------------------------------------------
    def _fill_fields(self, data) -> None:
        p = self.panel
        p.current_index = self.index
        p.current_type = self.conn_type
        p._populate_node_lists()
        p.type_combo.setCurrentIndex(0 if self.conn_type == "edge" else 1)
        if self.conn_type == "edge":
            p.source = data.get("from")
            p.target = data.get("to")
            p.source_edit.setCurrentText(p.source or "")
            p.target_edit.setCurrentText(p.target or "")
            p.atten_spin.setValue(float(data.get("attenuation", 1.0)))
            p.density_spin.setValue(float(data.get("density", 0.0)))
            p.delay_spin.setValue(float(data.get("delay", 1.0)))
            p.phase_shift_spin.setValue(float(data.get("phase_shift", 0.0)))
            p.weight_spin.setValue(float(data.get("weight", 0.0)))
        else:
            nodes = data.get("nodes", ["", ""])
            p.source, p.target = nodes[0], nodes[1]
            p.nodea_edit.setCurrentText(p.source)
            p.nodeb_edit.setCurrentText(p.target)
            p.bridge_type_combo.setCurrentText(data.get("bridge_type", "Braided"))
            p.phase_offset_spin.setValue(float(data.get("phase_offset", 0.0)))
            p.drift_tol_spin.setValue(float(data.get("drift_tolerance", 0.0)))
            p.decoherence_spin.setValue(float(data.get("decoherence_limit", 0.0)))
            p.initial_strength_spin.setValue(float(data.get("initial_strength", 0.0)))
            p.medium_type_edit.setText(data.get("medium_type", ""))
            p.mutable_check.setChecked(bool(data.get("mutable", False)))
            p.entangled_check.setChecked(bool(data.get("is_entangled", False)))
            p.entangled_id_label.setText(str(data.get("entangled_id", "")))

    # ------------------------------------------------------------------
    def _finish(self) -> None:
        self.panel._update_fields()
        self.panel.dirty = False
        self.panel.show()
