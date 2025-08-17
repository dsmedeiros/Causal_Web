from __future__ import annotations

"""Expose DOE queue results to QML panels."""

from typing import Dict, List, Optional

from PySide6.QtCore import QObject, Property, Signal, Slot

from experiments import DOEQueueManager
from ..ipc import Client


class DOEModel(QObject):
    """Run DOE sweeps and expose Top-K and scatter results."""

    runsChanged = Signal()
    topKChanged = Signal()
    scatterChanged = Signal()
    parallelChanged = Signal()
    heatmapChanged = Signal()

    def __init__(self) -> None:
        super().__init__()
        self._runs: List[dict] = []
        self._topk: List[dict] = []
        self._scatter: List[List[float]] = []
        self._parallel: List[List[float]] = []
        self._heatmap: List[List[float]] = []
        self._client: Optional[Client] = None
        self._mgr: Optional[DOEQueueManager] = None
        self._base = {
            "W0": 1.0,
            "alpha_leak": 1.0,
            "lambda_decay": 1.0,
            "b": 1.0,
            "prob": 0.5,
        }
        self._groups = {"Delta_over_W0": (0.0, 1.0), "alpha_d_over_leak": (0.0, 1.0)}
        self._gates: List[int] = []

    # ------------------------------------------------------------------
    @Slot(int)
    def runLhs(self, samples: int = 10) -> None:
        """Generate ``samples`` Latin Hypercube points and evaluate them."""

        self._mgr = DOEQueueManager(
            self._base, self._gates, self._fitness, client=self._client
        )
        self._mgr.enqueue_lhs(self._groups, samples)
        if self._client is None:
            self._mgr.run_all()
            self._recompute()
        else:
            import asyncio

            async def _run() -> None:
                await self._mgr.run_all_ipc()

            self._ipc_task = asyncio.create_task(_run())

    def handle_status(self, msg: Dict) -> None:
        """Process an ``ExperimentStatus`` message from the engine."""

        if not self._mgr:
            return
        idx = msg.get("id")
        if idx is None or idx >= len(self._mgr.runs):
            return
        _, status = self._mgr.runs[idx]
        state = msg.get("state")
        if state:
            status.state = str(state)
        inv = msg.get("invariants")
        if inv is not None:
            status.invariants = dict(inv)
        fit = msg.get("fitness")
        if fit is not None:
            status.fitness = float(fit)
        if status.state in {"finished", "failed"}:
            self._recompute()

    def set_client(self, client: Client) -> None:
        """Attach a WebSocket ``client`` for engine communication."""

        self._client = client

    def _recompute(self) -> None:
        """Recompute result tables based on current run statuses."""

        if not self._mgr:
            return
        mgr = self._mgr
        self._runs = [{"config": cfg, "status": status} for cfg, status in mgr.runs]
        self._topk = sorted(
            (
                {"fitness": status.fitness or 0.0, **cfg}
                for cfg, status in mgr.runs
                if status.fitness is not None
            ),
            key=lambda r: r["fitness"],
            reverse=True,
        )[:5]
        names = list(self._groups.keys())
        if len(names) >= 2:
            self._scatter = [
                [cfg[names[0]], cfg[names[1]], status.fitness or 0.0]
                for cfg, status in mgr.runs
            ]
        else:
            self._scatter = []
        self._parallel = [
            [
                (cfg[n] - self._groups[n][0])
                / ((self._groups[n][1] - self._groups[n][0]) or 1.0)
                for n in names
            ]
            + [status.fitness or 0.0]
            for cfg, status in mgr.runs
        ]
        if len(names) >= 2:
            bins = 10
            grid = [[0.0 for _ in range(bins)] for _ in range(bins)]
            counts = [[0 for _ in range(bins)] for _ in range(bins)]
            for cfg, status in mgr.runs:
                x_range = self._groups[names[0]]
                y_range = self._groups[names[1]]
                x = int(
                    (cfg[names[0]] - x_range[0])
                    / ((x_range[1] - x_range[0]) or 1.0)
                    * bins
                )
                y = int(
                    (cfg[names[1]] - y_range[0])
                    / ((y_range[1] - y_range[0]) or 1.0)
                    * bins
                )
                x = min(max(x, 0), bins - 1)
                y = min(max(y, 0), bins - 1)
                grid[y][x] += status.fitness or 0.0
                counts[y][x] += 1
            for j in range(bins):
                for i in range(bins):
                    if counts[j][i]:
                        grid[j][i] /= counts[j][i]
                    else:
                        grid[j][i] = None
            self._heatmap = grid
        else:
            self._heatmap = []
        self.runsChanged.emit()
        self.topKChanged.emit()
        self.scatterChanged.emit()
        self.parallelChanged.emit()
        self.heatmapChanged.emit()

    # ------------------------------------------------------------------
    def _fitness(
        self,
        metrics: Dict[str, float],
        invariants: Dict[str, float],
        groups: Dict[str, float],
        toggles: Dict[str, int],
    ) -> float:
        return groups.get("Delta_over_W0", 0.0)

    # ------------------------------------------------------------------
    def _get_topk(self) -> List[dict]:
        return self._topk

    topK = Property("QVariant", _get_topk, notify=topKChanged)

    def _get_scatter(self) -> List[List[float]]:
        return self._scatter

    scatter = Property("QVariant", _get_scatter, notify=scatterChanged)

    def _get_parallel(self) -> List[List[float]]:
        return self._parallel

    parallel = Property("QVariant", _get_parallel, notify=parallelChanged)

    def _get_heatmap(self) -> List[List[float]]:
        return self._heatmap

    heatmap = Property("QVariant", _get_heatmap, notify=heatmapChanged)

    def _get_group_names(self) -> List[str]:
        return list(self._groups.keys())

    groupNames = Property("QVariant", _get_group_names, constant=True)
