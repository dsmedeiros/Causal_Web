from __future__ import annotations

import threading
import time
from typing import Optional

from PySide6.QtCore import QObject, Signal

from ..config import Config, EngineMode
from ..engine.engine_v2.adapter import EngineAdapter
from ..view import ViewSnapshot


class EngineWorker(QObject):
    """Background worker that advances the engine on a separate thread."""

    snapshot_ready = Signal()

    def __init__(self, adapter: EngineAdapter) -> None:
        """Initialise the worker with the engine ``adapter``."""
        super().__init__()
        self._adapter = adapter
        self._lock = threading.Lock()
        self._latest: Optional[ViewSnapshot] = None
        self._active = True

    def run(self) -> None:
        """Continuously step the engine while the simulation is running."""
        while self._active:
            with Config.state_lock:
                running = Config.is_running and Config.engine_mode == EngineMode.V2
            if running:
                self._adapter.step()
                snap = self._adapter.snapshot_for_ui()
                with self._lock:
                    self._latest = snap
                self.snapshot_ready.emit()
            else:
                time.sleep(0.01)

    def latest_snapshot(self) -> Optional[ViewSnapshot]:
        """Return the most recent :class:`ViewSnapshot` and clear the buffer."""
        with self._lock:
            snap = self._latest
            self._latest = None
            return snap

    def stop(self) -> None:
        """Stop the worker loop."""
        self._active = False
