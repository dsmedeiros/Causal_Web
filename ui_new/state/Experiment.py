from __future__ import annotations

"""Experiment model exposed to QML panels."""

import asyncio
from typing import Optional

from PySide6.QtCore import QObject, Property, Signal, Slot

from ..ipc import Client


class ExperimentModel(QObject):
    """Track experiment status and residual metrics."""

    statusChanged = Signal(str)
    residualChanged = Signal(float)

    def __init__(self) -> None:
        super().__init__()
        self._status = ""
        self._residual = 0.0
        self._client: Optional[Client] = None

    # ------------------------------------------------------------------
    def _get_status(self) -> str:
        return self._status

    def _set_status(self, value: str) -> None:
        if self._status != value:
            self._status = value
            self.statusChanged.emit(value)

    status = Property(str, _get_status, _set_status, notify=statusChanged)

    # ------------------------------------------------------------------
    def _get_residual(self) -> float:
        return self._residual

    def _set_residual(self, value: float) -> None:
        if self._residual != value:
            self._residual = value
            self.residualChanged.emit(value)

    residual = Property(float, _get_residual, _set_residual, notify=residualChanged)

    # ------------------------------------------------------------------
    def update(self, status: str, residual: float) -> None:
        """Convenience method to update both fields."""
        self._set_status(status)
        self._set_residual(residual)

    # ------------------------------------------------------------------
    def set_client(self, client: Client) -> None:
        """Attach a WebSocket ``client`` for sending control messages."""
        self._client = client

    # ------------------------------------------------------------------
    @Slot()
    def start(self) -> None:
        """Request the backend to start the experiment."""
        if self._client:
            asyncio.create_task(
                self._client.send({"ExperimentControl": {"action": "start"}})
            )

    @Slot()
    def pause(self) -> None:
        """Request the backend to pause the experiment."""
        if self._client:
            asyncio.create_task(
                self._client.send({"ExperimentControl": {"action": "pause"}})
            )

    @Slot()
    def resume(self) -> None:
        """Request the backend to resume the experiment."""
        if self._client:
            asyncio.create_task(
                self._client.send({"ExperimentControl": {"action": "resume"}})
            )

    @Slot()
    def reset(self) -> None:
        """Request the backend to reset the experiment state."""
        if self._client:
            asyncio.create_task(
                self._client.send({"ExperimentControl": {"action": "reset"}})
            )
