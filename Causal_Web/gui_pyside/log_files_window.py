from __future__ import annotations

"""Window for toggling individual log files on or off."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QMainWindow,
    QLabel,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..config import Config
from .shared import TooltipCheckBox


LOG_TIPS: dict[str, str] = {
    "boundary_interaction_log.json": "Interactions with void or boundary nodes.",
    "bridge_decay_log.json": "Gradual weakening of inactive bridges.",
    "bridge_dynamics_log.json": "State changes for each bridge.",
    "bridge_reformation_log.json": "Bridges reforming after rupture.",
    "bridge_rupture_log.json": "Details of bridge failures.",
    "bridge_state.json": "Snapshot of all bridge states per tick.",
    "cluster_log.json": "Hierarchical clustering results.",
    "coherence_log.json": "Node coherence values.",
    "coherence_velocity_log.json": "Change in coherence between frames.",
    "collapse_chain_log.json": "Propagation chains triggered by collapse.",
    "collapse_front_log.json": "First collapse frame for each node.",
    "connectivity_log.json": "Number of links per node at load time.",
    "curvature_log.json": "Delay adjustments from law-wave curvature.",
    "decoherence_log.json": "Node decoherence levels.",
    "inspection_log.json": "Superposition inspection summary.",
    "interference_log.json": "Interference classification per node.",
    "interpretation_log.json": "Aggregated metrics from the interpreter.",
    "law_drift_log.json": "Refractory period adjustments.",
    "law_wave_log.json": "Node law-wave frequencies.",
    "layer_transition_log.json": "Frame transitions between LCCM layers.",
    "magnitude_failure_log.json": "Frames rejected for low magnitude.",
    "meta_node_ticks.json": "Frames emitted by meta nodes.",
    "node_emergence_log.json": "New nodes created via propagation.",
    "node_state_log.json": "Node type, credit and debt metrics.",
    "observer_disagreement_log.json": "Difference between observers and reality.",
    "propagation_failure_log.json": "Reasons frame propagation failed.",
    "proper_time_log.json": "Cumulative subjective frames per node.",
    "refraction_log.json": "Rerouted frames through alternative paths.",
    "should_tick_log.json": "Results of frame emission checks.",
    "stable_frequency_log.json": "Nodes with converged law-wave frequency values.",
    "structural_growth_log.json": "Summary of SIP/CSP growth each frame.",
    "tick_delivery_log.json": "Incoming frame phases for each node.",
    "tick_drop_log.json": "Frames discarded before firing.",
    "tick_emission_log.json": "Frames emitted by nodes.",
    "tick_evaluation_log.json": "Evaluation results for each potential frame.",
    "tick_propagation_log.json": "Frames travelling across edges.",
    "tick_seed_log.json": "Activity injected by the seeder.",
}


class LogFilesWindow(QMainWindow):
    """Window listing all known log files with enable checkboxes."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Log Files")
        self.resize(300, 400)

        self._checkboxes: dict[tuple[str, str], TooltipCheckBox] = {}

        central = QWidget()
        layout = QVBoxLayout(central)
        btn_layout = QHBoxLayout()
        self.apply_button = QPushButton("Apply")
        self.apply_button.clicked.connect(self.apply_changes)
        btn_layout.addWidget(self.apply_button, alignment=Qt.AlignLeft)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        interval_row = QHBoxLayout()
        interval_label = QLabel("Log Interval")
        self.interval_spin = QSpinBox()
        self.interval_spin.setMinimum(1)
        self.interval_spin.setMaximum(1000)
        self.interval_spin.setValue(getattr(Config, "log_interval", 1))
        interval_row.addWidget(interval_label)
        interval_row.addWidget(self.interval_spin)
        interval_row.addStretch()
        layout.addLayout(interval_row)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        checks = QVBoxLayout(container)

        tick_logs = sorted(Config.log_files.get("tick", {}))
        phen_logs = sorted(Config.log_files.get("phenomena", {}))
        event_logs = sorted(Config.log_files.get("event", {}))

        for title, names in [
            ("Frame", tick_logs),
            ("Phenomena", phen_logs),
            ("Events", event_logs),
        ]:
            if not names:
                continue
            header = QLabel(title)
            header.setStyleSheet("font-weight: bold")
            checks.addWidget(header)
            for name in names:
                desc = LOG_TIPS.get(f"{name}.json", "")
                cb = TooltipCheckBox(name, desc)
                category = "event" if title == "Events" else title.lower()
                checked = Config.log_files.get(category, {}).get(name, True)
                cb.setChecked(checked)
                self._checkboxes[(category, name)] = cb
                checks.addWidget(cb)
            checks.addSpacing(10)
        checks.addStretch()

        scroll.setWidget(container)
        layout.addWidget(scroll)
        self.setCentralWidget(central)

    def apply_changes(self) -> None:
        """Update :class:`Config.log_files` and write to ``config.json``."""
        for (category, label), cb in self._checkboxes.items():
            Config.log_files[category][label] = cb.isChecked()
        Config.log_interval = int(self.interval_spin.value())
        Config.save_log_files()
