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
    "bridge_state_log.json": "Snapshot of all bridge states per tick.",
    "cluster_log.json": "Hierarchical clustering results.",
    "coherence_log.json": "Node coherence values.",
    "coherence_velocity_log.json": "Change in coherence between ticks.",
    "collapse_chain_log.json": "Propagation chains triggered by collapse.",
    "collapse_front_log.json": "First collapse tick for each node.",
    "connectivity_log.json": "Number of links per node at load time.",
    "curvature_log.json": "Delay adjustments from law-wave curvature.",
    "decoherence_log.json": "Node decoherence levels.",
    "event_log.json": "High level events such as bridge ruptures.",
    "inspection_log.json": "Superposition inspection summary.",
    "interference_log.json": "Interference classification per node.",
    "interpretation_log.json": "Aggregated metrics from the interpreter.",
    "law_drift_log.json": "Refractory period adjustments.",
    "law_wave_log.json": "Node law-wave frequencies.",
    "layer_transition_log.json": "Tick transitions between LCCM layers.",
    "magnitude_failure_log.json": "Ticks rejected for low magnitude.",
    "meta_node_tick_log.json": "Ticks emitted by meta nodes.",
    "node_emergence_log.json": "New nodes created via propagation.",
    "node_state_log.json": "Node type, credit and debt metrics.",
    "observer_disagreement_log.json": "Difference between observers and reality.",
    "propagation_failure_log.json": "Reasons tick propagation failed.",
    "proper_time_log.json": "Cumulative subjective ticks per node.",
    "refraction_log.json": "Rerouted ticks through alternative paths.",
    "should_tick_log.json": "Results of tick emission checks.",
    "stable_frequency_log.json": "Nodes with converged law-wave frequency values.",
    "structural_growth_log.json": "Summary of SIP/CSP growth each tick.",
    "tick_delivery_log.json": "Incoming tick phases for each node.",
    "tick_drop_log.json": "Ticks discarded before firing.",
    "tick_emission_log.json": "Ticks emitted by nodes.",
    "tick_evaluation_log.json": "Evaluation results for each potential tick.",
    "tick_propagation_log.json": "Ticks travelling across edges.",
    "tick_seed_log.json": "Activity injected by the seeder.",
}


class LogFilesWindow(QMainWindow):
    """Window listing all known log files with enable checkboxes."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Log Files")
        self.resize(300, 400)

        self._checkboxes: dict[str, TooltipCheckBox] = {}

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

        tick_logs = sorted(n for n in Config.log_files if n in Config.TICK_FILES)
        phen_logs = sorted(n for n in Config.log_files if n in Config.PHENOMENA_FILES)
        event_logs = sorted(
            n
            for n in Config.log_files
            if n not in Config.TICK_FILES and n not in Config.PHENOMENA_FILES
        )

        for title, names in [
            ("Tick", tick_logs),
            ("Phenomena", phen_logs),
            ("Events", event_logs),
        ]:
            if not names:
                continue
            header = QLabel(title)
            header.setStyleSheet("font-weight: bold")
            checks.addWidget(header)
            for name in names:
                desc = LOG_TIPS.get(name, "")
                cb = TooltipCheckBox(name, desc)
                cb.setChecked(Config.log_files.get(name, True))
                self._checkboxes[name] = cb
                checks.addWidget(cb)
            checks.addSpacing(10)
        checks.addStretch()

        scroll.setWidget(container)
        layout.addWidget(scroll)
        self.setCentralWidget(central)

    def apply_changes(self) -> None:
        """Update :class:`Config.log_files` and write to ``config.json``."""
        for name, cb in self._checkboxes.items():
            Config.log_files[name] = cb.isChecked()
        Config.log_interval = int(self.interval_spin.value())
        Config.save_log_files()
