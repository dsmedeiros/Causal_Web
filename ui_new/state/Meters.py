"""Meters model reporting frame rate to QML."""

from __future__ import annotations

from PySide6.QtCore import QObject, Property, Signal, QElapsedTimer


class MetersModel(QObject):
    """Track frames per second for display in the Meters panel."""

    fpsChanged = Signal(float)

    def __init__(self) -> None:
        super().__init__()
        self._fps = 0.0
        self._frames = 0
        self._timer = QElapsedTimer()
        self._timer.start()

    # ------------------------------------------------------------------
    def frame_drawn(self) -> None:
        """Record that a frame was rendered and update FPS when needed."""

        self._frames += 1
        elapsed = self._timer.elapsed()
        if elapsed >= 1000:
            self._fps = self._frames * 1000.0 / elapsed
            self._frames = 0
            self._timer.restart()
            self.fpsChanged.emit(self._fps)

    # ------------------------------------------------------------------
    def _get_fps(self) -> float:
        return self._fps

    fps = Property(float, _get_fps, notify=fpsChanged)
