from __future__ import annotations

"""Expose the MCTS-H optimizer to QML panels."""

from pathlib import Path
from typing import Dict, List, Optional
import asyncio

from PySide6.QtCore import QObject, Property, Signal, Slot

from experiments import OptimizerQueueManager, MCTS_H, build_priors
from experiments.artifacts import load_hall_of_fame, load_top_k, write_best_config
from ..ipc import Client


class MCTSModel(QObject):
    """Run MCTS-H searches and surface hall-of-fame results."""

    hallOfFameChanged = Signal()
    runningChanged = Signal()
    baselinePromoted = Signal(str)
    statsChanged = Signal()

    def __init__(self) -> None:
        super().__init__()
        self._base = {
            "W0": 1.0,
            "alpha_leak": 1.0,
            "lambda_decay": 1.0,
            "b": 1.0,
            "prob": 0.5,
        }
        self._groups = {"Delta_over_W0": (0.0, 1.0)}
        self._gates: List[int] = [1]
        self._client: Optional[Client] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._mgr: Optional[OptimizerQueueManager] = None
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._c_ucb = 1.0
        self._alpha_pw = 0.5
        self._k_pw = 2.0
        self._promote = 0.2
        self._promote_q = 0.0
        self._promote_w = 0.0
        self._bins = 3
        self._multi_objective = False
        self._node_count = 0
        self._proxy_evals = 0
        self._full_evals = 0
        self._frontier = 0
        self._expansion_rate = 0.0
        self._max_nodes = 10000
        self._proxy_frames = 300
        self._full_frames = 3000
        data = load_hall_of_fame(Path("experiments/hall_of_fame.json"))
        self._hof: List[dict] = list(data.get("archive", []))

    # ------------------------------------------------------------------
    def set_client(
        self, client: Optional[Client], loop: Optional[asyncio.AbstractEventLoop] = None
    ) -> None:
        """Bind an IPC client and event loop for asynchronous execution."""

        self._client = client
        self._loop = loop

    # ------------------------------------------------------------------
    def handle_status(self, msg: Dict[str, object]) -> None:  # pragma: no cover - no-op
        """Consume experiment status messages from the engine."""

        del msg

    # ------------------------------------------------------------------
    @Slot()
    def start(self) -> None:
        """Begin a search using the configured parameters."""

        if self._running:
            return
        topk = load_top_k(Path("experiments/top_k.json"))
        priors = build_priors(
            [row["groups"] for row in topk.get("rows", [])], bins=self._bins
        )
        cfg = {
            "c_ucb": self._c_ucb,
            "alpha_pw": self._alpha_pw,
            "k_pw": self._k_pw,
            "multi_objective": self._multi_objective,
            "max_nodes": self._max_nodes,
        }
        if self._promote_q > 0.0:
            cfg["promote_quantile"] = self._promote_q
        else:
            cfg["promote_threshold"] = self._promote
        if self._promote_w > 0.0:
            cfg["promote_window"] = int(self._promote_w)
        opt = MCTS_H(list(self._groups.keys()), priors, cfg)
        self._mgr = OptimizerQueueManager(
            self._base,
            self._gates,
            self._fitness,
            opt,
            proxy_frames=self._proxy_frames,
            full_frames=self._full_frames,
        )
        self._node_count = 0
        self._proxy_evals = 0
        self._full_evals = 0
        self._running = True
        self.runningChanged.emit()
        self.statsChanged.emit()

        async def _run() -> None:
            while self._running and self._mgr:
                res = self._mgr.run_next()
                if res is None:
                    break
                if res.status == "proxy":
                    self._proxy_evals += 1
                elif res.status == "full":
                    self._full_evals += 1
                    data = load_hall_of_fame(Path("experiments/hall_of_fame.json"))
                    self._hof = list(data.get("archive", []))
                    self.hallOfFameChanged.emit()
                if self._mgr and getattr(self._mgr, "optimizer", None) is not None:
                    opt = self._mgr.optimizer
                    self._node_count = getattr(opt, "_nodes", self._node_count)
                    metrics = opt.metrics()
                    self._expansion_rate = float(
                        metrics.get("expansion_rate", self._expansion_rate)
                    )
                    self._frontier = int(metrics.get("frontier", self._frontier))
                self.statsChanged.emit()
                await asyncio.sleep(0)
            self._running = False
            self.runningChanged.emit()

        loop = self._loop or asyncio.get_event_loop()
        self._task = loop.create_task(_run())

    # ------------------------------------------------------------------
    @Slot()
    def pause(self) -> None:
        """Stop iterating the search loop."""

        self._running = False
        self.runningChanged.emit()

    # ------------------------------------------------------------------
    @Slot()
    def resume(self) -> None:
        """Resume a previously paused search."""

        if self._running or not self._mgr:
            return
        self._running = True
        self.runningChanged.emit()

        async def _run() -> None:
            while self._running and self._mgr:
                res = self._mgr.run_next()
                if res is None:
                    break
                if res.status == "proxy":
                    self._proxy_evals += 1
                elif res.status == "full":
                    self._full_evals += 1
                    data = load_hall_of_fame(Path("experiments/hall_of_fame.json"))
                    self._hof = list(data.get("archive", []))
                    self.hallOfFameChanged.emit()
                if self._mgr and getattr(self._mgr, "optimizer", None) is not None:
                    opt = self._mgr.optimizer
                    self._node_count = getattr(opt, "_nodes", self._node_count)
                    metrics = opt.metrics()
                    self._expansion_rate = float(
                        metrics.get("expansion_rate", self._expansion_rate)
                    )
                    self._frontier = int(metrics.get("frontier", self._frontier))
                self.statsChanged.emit()
                await asyncio.sleep(0)
            self._running = False
            self.runningChanged.emit()

        loop = self._loop or asyncio.get_event_loop()
        self._task = loop.create_task(_run())

    # ------------------------------------------------------------------
    @Slot()
    def promoteBaseline(self) -> None:
        """Persist the best configuration to ``best_config.yaml``."""

        data = load_top_k(Path("experiments/top_k.json"))
        rows = data.get("rows", [])
        if not rows:
            return
        path = write_best_config(rows[0])
        self.baselinePromoted.emit(path)

    # ------------------------------------------------------------------
    def _fitness(
        self,
        metrics: Dict[str, float],
        invariants: Dict[str, float],
        groups: Dict[str, float],
    ) -> float:
        """Toy fitness: optimise ``Delta_over_W0`` towards 0.0."""

        return groups.get("Delta_over_W0", 0.0)

    # ------------------------------------------------------------------
    def _get_hof(self) -> List[dict]:
        return self._hof

    hallOfFame = Property("QVariant", _get_hof, notify=hallOfFameChanged)

    def _get_running(self) -> bool:
        return self._running

    running = Property(bool, _get_running, notify=runningChanged)

    def _get_c_ucb(self) -> float:
        return self._c_ucb

    def _set_c_ucb(self, val: float) -> None:
        self._c_ucb = float(val)

    cUcb = Property(float, _get_c_ucb, _set_c_ucb)

    def _get_alpha_pw(self) -> float:
        return self._alpha_pw

    def _set_alpha_pw(self, val: float) -> None:
        self._alpha_pw = float(val)

    alphaPw = Property(float, _get_alpha_pw, _set_alpha_pw)

    def _get_k_pw(self) -> float:
        return self._k_pw

    def _set_k_pw(self, val: float) -> None:
        self._k_pw = float(val)

    kPw = Property(float, _get_k_pw, _set_k_pw)

    def _get_promote(self) -> float:
        return self._promote

    def _set_promote(self, val: float) -> None:
        self._promote = float(val)

    promoteThreshold = Property(float, _get_promote, _set_promote)

    def _get_promote_q(self) -> float:
        return self._promote_q

    def _set_promote_q(self, val: float) -> None:
        self._promote_q = float(val)

    promoteQuantile = Property(float, _get_promote_q, _set_promote_q)

    def _get_promote_w(self) -> float:
        return self._promote_w

    def _set_promote_w(self, val: float) -> None:
        self._promote_w = float(val)

    promoteWindow = Property(float, _get_promote_w, _set_promote_w)

    def _get_max_nodes(self) -> int:
        return self._max_nodes

    def _set_max_nodes(self, val: int) -> None:
        self._max_nodes = int(val)

    maxNodes = Property(int, _get_max_nodes, _set_max_nodes)

    def _get_proxy_frames(self) -> int:
        return self._proxy_frames

    def _set_proxy_frames(self, val: int) -> None:
        self._proxy_frames = int(val)

    proxyFrames = Property(int, _get_proxy_frames, _set_proxy_frames)

    def _get_full_frames(self) -> int:
        return self._full_frames

    def _set_full_frames(self, val: int) -> None:
        self._full_frames = int(val)

    fullFrames = Property(int, _get_full_frames, _set_full_frames)

    def _get_bins(self) -> int:
        return self._bins

    def _set_bins(self, val: int) -> None:
        self._bins = int(val)

    bins = Property(int, _get_bins, _set_bins)

    def _get_multi(self) -> bool:
        return self._multi_objective

    def _set_multi(self, val: bool) -> None:
        self._multi_objective = bool(val)

    multiObjective = Property(bool, _get_multi, _set_multi)

    def _get_node_count(self) -> int:
        return self._node_count

    def _get_proxy(self) -> int:
        return self._proxy_evals

    def _get_full(self) -> int:
        return self._full_evals

    def _get_promotion_rate(self) -> float:
        if self._proxy_evals == 0:
            return 0.0
        return self._full_evals / self._proxy_evals

    def _get_frontier(self) -> int:
        return self._frontier

    def _get_expansion_rate(self) -> float:
        return self._expansion_rate

    nodeCount = Property(int, _get_node_count, notify=statsChanged)
    proxyEvaluations = Property(int, _get_proxy, notify=statsChanged)
    fullEvaluations = Property(int, _get_full, notify=statsChanged)
    promotionRate = Property(float, _get_promotion_rate, notify=statsChanged)
    frontier = Property(int, _get_frontier, notify=statsChanged)
    expansionRate = Property(float, _get_expansion_rate, notify=statsChanged)
