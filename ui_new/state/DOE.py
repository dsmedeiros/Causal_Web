from __future__ import annotations

"""Expose DOE queue results to QML panels."""

from typing import Dict, List, Optional, Tuple
from pathlib import Path
import asyncio
import time

from PySide6.QtCore import QObject, Property, Signal, Slot

from experiments import DOEQueueManager
from ..ipc import Client
from experiments.artifacts import (
    TopKEntry,
    update_top_k,
    load_top_k,
    persist_run,
    allocate_run_dir,
    write_best_config,
)


class DOEModel(QObject):
    """Run DOE sweeps and expose Top-K and scatter results."""

    runsChanged = Signal()
    topKChanged = Signal()
    scatterChanged = Signal()
    parallelChanged = Signal()
    heatmapChanged = Signal()
    rangesChanged = Signal()
    progressChanged = Signal()
    etaChanged = Signal()
    brushesChanged = Signal()
    groupNamesChanged = Signal()
    baselinePromoted = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self._runs: List[dict] = []
        self._topk: List[dict] = []
        self._scatter: List[List[float]] = []
        self._parallel: List[List[float]] = []
        self._heatmap: List[List[float]] = []
        self._metric_names: List[str] = []
        self._all_rows: List[dict] = []
        self._all_scatter: List[List[float]] = []
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
        self._ipc_task: Optional[asyncio.Task] = None
        self._start_time: float | None = None
        self._progress: float = 0.0
        self._eta: float = 0.0
        self._steps = {n: 3 for n in self._groups}
        self._brushes: Dict[int, Tuple[float, float]] = {}
        data = load_top_k(Path("experiments/top_k.json"))
        self._topk = data.get("rows", [])

    # ------------------------------------------------------------------
    @Slot(int, bool)
    def runLhs(self, samples: int = 10, force: bool = False) -> None:
        """Generate ``samples`` LHS points and evaluate them.

        Parameters
        ----------
        samples:
            Number of Latin Hypercube samples to generate.
        force:
            When ``True`` re-evaluate configurations even if present in the
            run index.
        """

        self._mgr = DOEQueueManager(
            self._base, self._gates, self._fitness, client=self._client
        )
        self._mgr.enqueue_lhs(self._groups, samples, force=force)
        self._start_time = time.time()
        self._update_progress()
        if self._client is None:
            self._mgr.run_all()
            self._recompute()
        else:

            async def _run() -> None:
                await self._mgr.run_all_ipc()
                self._update_progress()

            self._ipc_task = asyncio.create_task(_run())

    # ------------------------------------------------------------------
    @Slot(bool)
    def runGrid(self, force: bool = False) -> None:
        """Execute a grid sweep using the configured ``steps`` per group.

        Parameters
        ----------
        force:
            When ``True`` re-evaluate configurations even if present in the
            run index.
        """

        self._mgr = DOEQueueManager(
            self._base, self._gates, self._fitness, client=self._client
        )
        self._mgr.enqueue_grid(self._groups, self._steps, force=force)
        self._start_time = time.time()
        self._update_progress()
        if self._client is None:
            self._mgr.run_all()
            self._recompute()
        else:

            async def _run() -> None:
                await self._mgr.run_all_ipc()
                self._update_progress()

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
        rows: List[dict] = []
        entries: List[TopKEntry] = []
        metric_keys: set[str] = set()
        for cfg, status in mgr.runs:
            if status.fitness is None:
                continue
            if status.path is None:
                rid, abs_path, rel_path = allocate_run_dir()
                persist_run(
                    cfg,
                    {"fitness": status.fitness, "invariants": status.invariants or {}},
                    abs_path,
                )
                status.run_id = rid
                status.path = rel_path
            row = {
                "fitness": status.fitness or 0.0,
                "path": status.path,
                "run_id": status.run_id or "",
                "seed": 0,
                "groups": cfg,
                "toggles": {},
                **cfg,
            }
            rows.append(row)
            metric_keys.update(
                k
                for k, v in (status.invariants or {}).items()
                if isinstance(v, (int, float))
            )
            entries.append(
                TopKEntry(
                    run_id=status.run_id or "",
                    fitness=status.fitness or 0.0,
                    objectives={
                        k: float(v)
                        for k, v in (status.invariants or {}).items()
                        if isinstance(v, (int, float))
                    },
                    groups=cfg,
                    toggles={},
                    seed=0,
                    path=status.path or "",
                )
            )
        self._all_rows = rows
        update_top_k(entries, Path("experiments/top_k.json"))
        names = list(self._groups.keys())
        if len(names) >= 2:
            scatter = [
                [cfg[names[0]], cfg[names[1]], status.fitness or 0.0]
                for cfg, status in mgr.runs
            ]
        else:
            scatter = []
        self._all_scatter = scatter
        metric_keys = sorted(metric_keys)
        self._metric_names = list(metric_keys)
        mmins = {k: float("inf") for k in metric_keys}
        mmaxs = {k: float("-inf") for k in metric_keys}
        for _, status in mgr.runs:
            inv = status.invariants or {}
            for k in metric_keys:
                val = inv.get(k)
                if isinstance(val, (int, float)):
                    if val < mmins[k]:
                        mmins[k] = float(val)
                    if val > mmaxs[k]:
                        mmaxs[k] = float(val)
        fits = [s.fitness for _, s in mgr.runs if s.fitness is not None]
        fmin, fmax = (min(fits), max(fits)) if fits else (0.0, 1.0)
        self._parallel = []
        for cfg, status in mgr.runs:
            gvals = [
                (cfg[n] - self._groups[n][0])
                / ((self._groups[n][1] - self._groups[n][0]) or 1.0)
                for n in names
            ]
            inv = status.invariants or {}
            mvals = [
                (
                    (float(inv.get(k, mmins[k])) - mmins[k])
                    / ((mmaxs[k] - mmins[k]) or 1.0)
                )
                for k in metric_keys
            ]
            fnorm = ((status.fitness or 0.0) - fmin) / ((fmax - fmin) or 1.0)
            self._parallel.append(gvals + mvals + [fnorm])
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
        self.parallelChanged.emit()
        self.heatmapChanged.emit()
        self.groupNamesChanged.emit()
        self._apply_brush_filters()
        self._update_progress()

    # ------------------------------------------------------------------
    def _apply_brush_filters(self) -> None:
        """Filter Top-K and scatter based on current brushes."""

        if not self._all_rows:
            self._topk = []
            self._scatter = []
            self.topKChanged.emit()
            self.scatterChanged.emit()
            return
        if not self._brushes:
            rows = sorted(self._all_rows, key=lambda r: r["fitness"], reverse=True)
            self._topk = rows[:5]
            self._scatter = list(self._all_scatter)
            self.topKChanged.emit()
            self.scatterChanged.emit()
            return
        idxs: List[int] = []
        for i, vec in enumerate(self._parallel):
            keep = True
            for ax, (low, high) in self._brushes.items():
                val = vec[ax]
                if val < low or val > high:
                    keep = False
                    break
            if keep:
                idxs.append(i)
        rows = [self._all_rows[i] for i in idxs]
        rows.sort(key=lambda r: r["fitness"], reverse=True)
        self._topk = rows[:5]
        self._scatter = [self._all_scatter[i] for i in idxs]
        self.topKChanged.emit()
        self.scatterChanged.emit()

    # ------------------------------------------------------------------
    @Slot(str, float, float)
    def setGroupRange(self, name: str, low: float, high: float) -> None:
        """Update the ``low``/``high`` range for the given ``name``."""

        self._groups[name] = (float(low), float(high))
        self.rangesChanged.emit()

    # ------------------------------------------------------------------
    @Slot(str, int)
    def setGroupSteps(self, name: str, steps: int) -> None:
        """Set grid sweep ``steps`` for ``name``."""

        self._steps[name] = int(steps)
        self.rangesChanged.emit()

    # ------------------------------------------------------------------
    def _update_progress(self) -> None:
        """Recompute progress ratio and ETA based on run states."""

        if not self._mgr:
            self._progress = 0.0
            self._eta = 0.0
        else:
            total = len(self._mgr.runs)
            done = sum(
                1 for _, s in self._mgr.runs if s.state in {"finished", "failed"}
            )
            self._progress = done / total if total else 0.0
            if self._start_time and done:
                elapsed = time.time() - self._start_time
                remaining = (elapsed / done) * (total - done)
                self._eta = remaining
            else:
                self._eta = 0.0
        self.progressChanged.emit()
        self.etaChanged.emit()

    # ------------------------------------------------------------------
    @Slot()
    def stop(self) -> None:
        """Cancel any in-flight sweep."""

        if self._ipc_task is not None:
            self._ipc_task.cancel()
            self._ipc_task = None

    # ------------------------------------------------------------------
    @Slot()
    def resume(self) -> None:
        """Resume a previously stopped sweep."""

        if not self._mgr or self._client is None:
            return
        if self._ipc_task is None:
            self._start_time = self._start_time or time.time()

            async def _run() -> None:
                await self._mgr.run_all_ipc()
                self._update_progress()

            self._ipc_task = asyncio.create_task(_run())

    # ------------------------------------------------------------------
    @Slot(int, float, float)
    def setBrush(self, axis: int, low: float, high: float) -> None:
        """Apply a brush ``(low, high)`` on ``axis`` in normalised units."""

        if low >= high:
            self._brushes.pop(axis, None)
        else:
            self._brushes[axis] = (max(0.0, low), min(1.0, high))
        self.brushesChanged.emit()
        self._apply_brush_filters()

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
    @Slot("QVariant")
    def promote(self, row: dict) -> None:
        """Write ``row`` configuration to ``best_config.yaml``."""
        if isinstance(row, dict):
            path = write_best_config(row)
            self.baselinePromoted.emit(path)

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
        return list(self._groups.keys()) + self._metric_names + ["fitness"]

    groupNames = Property("QVariant", _get_group_names, notify=groupNamesChanged)

    def _get_groups(self) -> List[dict]:
        return [
            {"name": n, "low": rng[0], "high": rng[1], "steps": self._steps.get(n, 1)}
            for n, rng in self._groups.items()
        ]

    groups = Property("QVariant", _get_groups, notify=rangesChanged)

    def _get_progress(self) -> float:
        return self._progress

    progress = Property(float, _get_progress, notify=progressChanged)

    def _get_eta(self) -> float:
        return self._eta

    eta = Property(float, _get_eta, notify=etaChanged)

    def _get_brushes(self) -> Dict[int, Tuple[float, float]]:
        return self._brushes

    brushes = Property("QVariant", _get_brushes, notify=brushesChanged)
