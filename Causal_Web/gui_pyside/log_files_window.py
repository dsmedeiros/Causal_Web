from __future__ import annotations

"""Window for toggling individual log files on or off."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
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


class LogFilesWindow(QMainWindow):
    """Window listing all known log files with enable checkboxes."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Log Files")
        self.resize(300, 400)

        self._checkboxes: dict[str, QCheckBox] = {}

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

        for name in sorted(Config.log_files):
            cb = QCheckBox(name)
            cb.setChecked(Config.log_files.get(name, True))
            self._checkboxes[name] = cb
            checks.addWidget(cb)
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
