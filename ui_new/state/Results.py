from __future__ import annotations

"""Expose results registry queries to QML panels."""

import asyncio
from pathlib import Path
from typing import List, Dict, Any

from PySide6.QtCore import QObject, Property, Signal, Slot

from experiments import results_registry as rr
from .Replay import open_replay


class ResultsModel(QObject):
    """Filter and retrieve experiment results for display."""

    rowsChanged = Signal()
    filtersChanged = Signal()

    def __init__(self) -> None:
        super().__init__()
        db_path = Path("experiments/results.db")
        self._conn = rr.connect(db_path)
        self._optimizer = "mcts_h"
        self._promotion = 0.0
        self._corr = 0.0
        self._rows: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    def _get_optimizer(self) -> str:
        return self._optimizer

    def _set_optimizer(self, val: str) -> None:
        self._optimizer = val
        self.refresh()
        self.filtersChanged.emit()

    optimizer = Property(str, _get_optimizer, _set_optimizer, notify=filtersChanged)

    def _get_promotion(self) -> float:
        return self._promotion

    def _set_promotion(self, val: float) -> None:
        self._promotion = float(val)
        self.refresh()
        self.filtersChanged.emit()

    promotionMin = Property(
        float, _get_promotion, _set_promotion, notify=filtersChanged
    )

    def _get_corr(self) -> float:
        return self._corr

    def _set_corr(self, val: float) -> None:
        self._corr = float(val)
        self.refresh()
        self.filtersChanged.emit()

    proxyFullCorrMin = Property(float, _get_corr, _set_corr, notify=filtersChanged)

    def _get_rows(self) -> List[Dict[str, Any]]:
        return self._rows

    rows = Property("QVariant", _get_rows, notify=rowsChanged)

    # ------------------------------------------------------------------
    @Slot()
    def refresh(self) -> None:
        """Query the registry with current filters."""

        rows = rr.query(
            self._conn,
            optimizer=self._optimizer or None,
            promotion_min=self._promotion,
            proxy_full_corr_min=self._corr,
        )
        self._rows = [dict(r) for r in rows]
        self.rowsChanged.emit()

    # ------------------------------------------------------------------
    @Slot(str)
    def openReplay(self, run_id: str) -> None:
        """Load ``run_id`` into the Replay panel and start playback."""

        asyncio.create_task(open_replay(run_id))
