from __future__ import annotations

"""Replay model exposed to QML panels."""

import asyncio
from typing import Optional

from PySide6.QtCore import QObject, Property, Signal, Slot

from ..ipc import Client


class ReplayModel(QObject):
    """Track replay progress as a fraction [0, 1]."""

    progressChanged = Signal(float)

    def __init__(self) -> None:
        super().__init__()
        self._progress = 0.0
        self._client: Optional[Client] = None

    # ------------------------------------------------------------------
    def _get_progress(self) -> float:
        return self._progress

    def _set_progress(self, value: float) -> None:
        if self._progress != value:
            self._progress = value
            self.progressChanged.emit(value)

    progress = Property(float, _get_progress, _set_progress, notify=progressChanged)

    # ------------------------------------------------------------------
    def update_progress(self, value: float) -> None:
        """Set the current replay progress."""
        self._set_progress(value)

    # ------------------------------------------------------------------
    def set_client(self, client: Client) -> None:
        """Attach a WebSocket ``client`` for sending control messages."""
        self._client = client

    # ------------------------------------------------------------------
    @Slot()
    def play(self) -> None:
        """Start or resume the replay."""
        if self._client:
            asyncio.create_task(
                self._client.send({"ReplayControl": {"action": "play"}})
            )

    @Slot()
    def pause(self) -> None:
        """Pause the replay."""
        if self._client:
            asyncio.create_task(
                self._client.send({"ReplayControl": {"action": "pause"}})
            )

    @Slot(float)
    def seek(self, value: float) -> None:
        """Seek the replay to ``value`` between 0 and 1."""
        self._set_progress(value)
        if self._client:
            asyncio.create_task(
                self._client.send(
                    {"ReplayControl": {"action": "seek", "progress": float(value)}}
                )
            )
