"""Shared Qt helpers for the GUI."""

from __future__ import annotations

from pathlib import Path
import json
from typing import Dict, Callable

from PySide6.QtCore import QObject, QEvent, QTimer
from PySide6.QtWidgets import QLabel

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

    def enterEvent(self, event: QEvent) -> None:  # type: ignore[override]
        if self.tip:
            self._timer.start(500)
        super().enterEvent(event)

    def leaveEvent(self, event: QEvent) -> None:  # type: ignore[override]
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

    def __init__(self, callback: Callable[[], None]) -> None:
        super().__init__()
        self.callback = callback

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:  # type: ignore[override]
        if event.type() == QEvent.FocusOut:
            self.callback()
        return False
