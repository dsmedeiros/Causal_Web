from __future__ import annotations

"""Dock widget displaying a snapshot of engine configuration."""

from PySide6.QtWidgets import QDockWidget, QWidget, QFormLayout, QLabel

from ..config import Config


class EngineProfileDock(QDockWidget):
    """Dock widget showing the current engine configuration."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Engine Profile", parent)
        self.setFeatures(QDockWidget.NoDockWidgetFeatures)
        self._container = QWidget()
        self._layout = QFormLayout(self._container)
        self.setWidget(self._container)
        self._labels: dict[str, QLabel] = {}
        self.update_snapshot()

    def update_snapshot(self) -> None:
        """Refresh labels with the latest configuration values."""
        snap = Config.snapshot()
        fields = {
            "Engine Mode": getattr(Config.engine_mode, "value", "unknown"),
            "Frame Rate": getattr(Config, "tick_rate", 0),
            "Frame Limit": getattr(Config, "max_ticks", 0),
            "Run Seed": snap.run_seed,
        }
        for key, value in fields.items():
            if key not in self._labels:
                lbl = QLabel(str(value))
                self._layout.addRow(key, lbl)
                self._labels[key] = lbl
            else:
                self._labels[key].setText(str(value))
