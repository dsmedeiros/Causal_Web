from __future__ import annotations

"""Experiment model exposed to QML panels."""

import asyncio
from pathlib import Path
from typing import Optional

import yaml
from PySide6.QtCore import QObject, Property, Signal, Slot

from config.normalizer import Normalizer
from ..ipc import Client


class ExperimentModel(QObject):
    """Track experiment status and residual metrics."""

    statusChanged = Signal(str)
    residualChanged = Signal(float)
    rateChanged = Signal(float)

    def __init__(self) -> None:
        super().__init__()
        self._status = ""
        self._residual = 0.0
        self._client: Optional[Client] = None
        self._rate = 1.0

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
    def reset(self) -> None:
        """Request the backend to reset the experiment state."""
        if self._client:
            asyncio.create_task(
                self._client.send({"ExperimentControl": {"action": "reset"}})
            )

    @Slot()
    def step(self) -> None:
        """Request a single-step advancement of the experiment."""
        if self._client:
            asyncio.create_task(
                self._client.send({"ExperimentControl": {"action": "step"}})
            )

    @Slot(float)
    def setRate(self, value: float) -> None:
        """Set the desired simulation ``value`` speed multiplier."""
        if self._rate != value:
            self._rate = value
            self.rateChanged.emit(value)
        if self._client:
            asyncio.create_task(
                self._client.send(
                    {"ExperimentControl": {"action": "set_rate", "rate": value}}
                )
            )

    # ------------------------------------------------------------------
    @Slot()
    def runBaseline(self) -> None:
        """Execute a run using ``experiments/best_config.yaml``."""

        if self._client is None:
            return
        path = Path("experiments/best_config.yaml")
        if not path.exists():
            self._set_status(f"Baseline not found: {path}")
            return
        try:
            data = yaml.safe_load(path.read_text()) or {}
        except Exception as exc:
            self._set_status(f"Failed to load baseline: {exc}")
            return
        groups = data.get("dimensionless", {})
        toggles = data.get("toggles", {})
        seed = int(data.get("seed", 0))
        base = {
            "W0": 1.0,
            "alpha_leak": 1.0,
            "lambda_decay": 1.0,
            "b": 1.0,
            "prob": 0.5,
        }
        raw = Normalizer().to_raw(base, groups)
        raw.update(toggles)
        raw["seed"] = seed
        asyncio.create_task(
            self._client.send({"ExperimentControl": {"action": "run", "config": raw}})
        )

    # ------------------------------------------------------------------
    def _get_rate(self) -> float:
        return self._rate

    rate = Property(float, _get_rate, notify=rateChanged)
