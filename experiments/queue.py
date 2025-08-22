"""Simple DOE queue manager.

This module exposes :class:`DOEQueueManager` which prepares design of
experiments sweeps over dimensionless groups using either Latin Hypercube or
uniform grid sampling.  Each enqueued sample records a live status along with
invariant and optional fitness results once executed.
"""

from __future__ import annotations

from dataclasses import dataclass
import asyncio
import itertools
import json
import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from config.normalizer import Normalizer
from invariants import checks
from .gates import run_gates
from .runner import _latin_hypercube
from .index import RunIndex, run_key
from .artifacts import (
    TopKEntry,
    update_top_k,
    save_hall_of_fame,
    persist_run,
    allocate_run_dir,
)
from .optim import Optimizer, MCTS_H

logger = logging.getLogger(__name__)


@dataclass
class RunStatus:
    """Track the state of a single experiment run.

    ``run_id`` and ``path`` are populated once a run has been persisted to
    disk.  ``path`` is stored relative to the ``experiments`` directory so it
    can be embedded directly into Top-K artifacts.  ``force`` marks runs that
    should execute even when a matching configuration already exists in the
    run index.
    """

    state: str = "queued"
    invariants: Optional[Dict[str, float]] = None
    fitness: Optional[float] = None
    error: Optional[str] = None
    run_id: Optional[str] = None
    path: Optional[str] = None
    force: bool = False


class DOEQueueManager:
    """Manage a queue of DOE runs.

    Parameters
    ----------
    base:
        Baseline raw configuration used to materialise dimensionless groups.
    gates:
        Sequence of gate identifiers executed for each sample.
    fitness_fn:
        Optional callback returning a scalar fitness value given gate metrics,
        invariants and the groups and toggles used for the run.
    seed:
        Seed for deterministic sampling.
    run_index:
        Optional :class:`RunIndex` used to track completed runs and avoid
        duplicate evaluations.
    """

    def __init__(
        self,
        base: Dict[str, float],
        gates: Iterable[int],
        fitness_fn: (
            Callable[
                [Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, int]],
                float,
            ]
            | None
        ) = None,
        seed: int = 0,
        client: Any | None = None,
        run_index: RunIndex | None = None,
    ) -> None:
        self.base = dict(base)
        self.gates = list(gates)
        self.fitness_fn = fitness_fn
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self._runs: List[Tuple[Dict[str, float], RunStatus]] = []
        self._normalizer = Normalizer()
        self._client = client
        self._index = run_index or RunIndex()

    # ------------------------------------------------------------------
    # queue construction
    def enqueue_lhs(
        self,
        groups: Dict[str, Tuple[float, float]],
        samples: int,
        *,
        force: bool = False,
    ) -> None:
        """Enqueue samples via Latin Hypercube sampling.

        Parameters
        ----------
        groups:
            Mapping of group names to ``(low, high)`` ranges.
        samples:
            Number of samples to generate.
        force:
            When ``True`` enqueue runs even if already present in the index.
        """

        names = list(groups.keys())
        ranges = np.array([groups[n] for n in names], dtype=float)
        unit = _latin_hypercube(samples, len(names), self._rng)
        lows, highs = ranges[:, 0], ranges[:, 1]
        scaled = lows[None, :] + unit * (highs - lows)[None, :]
        for i in range(samples):
            cfg = dict(zip(names, scaled[i]))
            raw = self._normalizer.to_raw(self.base, cfg)
            seed = int(raw.get("seed", self.seed))
            key = run_key(
                {
                    "groups": cfg,
                    "toggles": {},
                    "seed": seed,
                    "gates": self.gates,
                }
            )
            if not force and key in self._index:
                continue
            self._runs.append((cfg, RunStatus(force=force)))

    def enqueue_grid(
        self,
        groups: Dict[str, Tuple[float, float]],
        steps: Dict[str, int],
        *,
        force: bool = False,
    ) -> None:
        """Enqueue a uniform grid sweep.

        Parameters
        ----------
        groups:
            Mapping of group names to ``(low, high)`` ranges.
        steps:
            Number of steps along each group dimension.
        force:
            When ``True`` enqueue runs even if already present in the index.
        """

        names = list(groups.keys())
        axes: List[np.ndarray] = []
        for n in names:
            low, high = groups[n]
            cnt = steps.get(n, 1)
            axes.append(np.linspace(low, high, cnt))
        for combo in itertools.product(*axes):
            cfg = dict(zip(names, combo))
            raw = self._normalizer.to_raw(self.base, cfg)
            seed = int(raw.get("seed", self.seed))
            key = run_key(
                {
                    "groups": cfg,
                    "toggles": {},
                    "seed": seed,
                    "gates": self.gates,
                }
            )
            if not force and key in self._index:
                continue
            self._runs.append((cfg, RunStatus(force=force)))

    # ------------------------------------------------------------------
    # execution
    def run_next(self) -> Optional[RunStatus]:
        """Execute the next queued run locally.

        The synchronous path is kept for tests and command-line utilities. When
        an IPC ``client`` is supplied use :meth:`run_next_ipc` to delegate
        execution to the running engine.

        Returns
        -------
        RunStatus or ``None`` if no runs remain.
        """

        for i, (groups, status) in enumerate(self._runs):
            if status.state == "queued":
                raw = self._normalizer.to_raw(self.base, groups)
                seed = int(raw.get("seed", self.seed))
                cfg = {
                    "groups": groups,
                    "toggles": {},
                    "seed": seed,
                    "gates": self.gates,
                }
                key = run_key(cfg)
                info = None if status.force else self._index.get(key)
                if info is not None:
                    status.state = "finished"
                    run_dir = self._index.runs_root / info
                    try:
                        res = json.loads((run_dir / "result.json").read_text())
                    except Exception:
                        res = {}
                    status.invariants = res.get("invariants")
                    status.fitness = res.get("fitness")
                    try:
                        manifest = json.loads((run_dir / "manifest.json").read_text())
                        status.run_id = manifest.get("run_id")
                    except Exception:
                        status.run_id = None
                    status.path = str(Path("runs") / info)
                    return status
                status.state = "running"
                try:
                    metrics = run_gates(raw, self.gates)
                    inv = checks.from_metrics(metrics)
                    status.invariants = inv
                    if self.fitness_fn is not None:
                        status.fitness = self.fitness_fn(metrics, inv, groups, {})
                    status.state = "finished"
                    rid, abs_path, rel_path = allocate_run_dir()
                    manifest = {
                        "run_id": rid,
                        "run_key": key,
                        "groups": groups,
                        "toggles": {},
                        "seed": seed,
                        "gates": self.gates,
                    }
                    res = {
                        "status": "ok",
                        "metrics": metrics,
                        "invariants": inv,
                        "fitness": status.fitness,
                    }
                    persist_run(raw, res, abs_path, manifest=manifest)
                    self._index.mark(key, rel_path)
                    status.run_id = rid
                    status.path = rel_path
                except Exception as exc:  # pragma: no cover - pass through
                    status.state = "failed"
                    status.error = str(exc)
                    raise
                return status
        return None

    async def run_next_ipc(self) -> Optional[RunStatus]:
        """Send the next queued run to the engine via the IPC client.

        The engine must respond with an ``ExperimentStatus`` message containing
        the matching ``id`` and resulting invariants/fitness.  This coroutine
        merely dispatches the configuration and marks the run as ``running``.
        """

        if self._client is None:
            return self.run_next()

        for i, (groups, status) in enumerate(self._runs):
            if status.state == "queued":
                raw = self._normalizer.to_raw(self.base, groups)
                seed = int(raw.get("seed", self.seed))
                cfg = {
                    "groups": groups,
                    "toggles": {},
                    "seed": seed,
                    "gates": self.gates,
                }
                key = run_key(cfg)
                info = None if status.force else self._index.get(key)
                if info is not None:
                    status.state = "finished"
                    run_dir = self._index.runs_root / info
                    try:
                        res = json.loads((run_dir / "result.json").read_text())
                    except Exception:
                        res = {}
                    status.invariants = res.get("invariants")
                    status.fitness = res.get("fitness")
                    try:
                        manifest = json.loads((run_dir / "manifest.json").read_text())
                        status.run_id = manifest.get("run_id")
                    except Exception:
                        status.run_id = None
                    status.path = str(Path("runs") / info)
                    return status
                status.state = "running"
                await self._client.send(
                    {"ExperimentControl": {"action": "run", "id": i, "config": raw}}
                )
                return status
        return None

    def run_all(self) -> List[RunStatus]:
        """Run all queued experiments sequentially using the local runner."""

        results: List[RunStatus] = []
        while True:
            res = self.run_next()
            if res is None:
                break
            results.append(res)
        return results

    async def run_all_ipc(self) -> List[RunStatus]:
        """Dispatch all queued runs to the engine sequentially.

        Each run is issued only after the previous one has completed.  Callers
        must process incoming ``ExperimentStatus`` messages to update the run
        states.
        """

        results: List[RunStatus] = []
        for _ in range(len(self._runs)):
            res = await self.run_next_ipc()
            if res is None:
                break
            results.append(res)
            while res.state == "running":
                await asyncio.sleep(0.05)
        return results

    @property
    def runs(self) -> List[Tuple[Dict[str, float], RunStatus]]:
        """Return queued runs with their status."""

        return list(self._runs)


@dataclass
class OptimizerResult:
    """Summary of a single optimizer-driven evaluation."""

    config: Dict[str, float]
    status: str
    fitness: Optional[float] = None
    path: Optional[str] = None


class OptimizerQueueManager:
    """Drive an :class:`~experiments.optim.Optimizer` using existing runners.

    This manager requests configurations from an optimizer, evaluates them via
    :func:`run_gates` and reports results back through
    :meth:`Optimizer.observe`.  Proxy evaluations can be promoted to full
    runs, in which case artifacts and the run index are updated.

    Parameters
    ----------
    base:
        Baseline raw configuration used to materialise groups.
    gates:
        Sequence of gate identifiers executed for each suggestion.
    fitness_fn:
        Callback returning a scalar fitness given metrics, invariants and the
        evaluated groups.  Higher fitness indicates a better configuration.
    optimizer:
        Optimizer instance implementing ``suggest`` and ``observe``.
    seed:
        Seed used when materialising raw configurations.
    proxy_frames:
        Number of frames used for initial proxy evaluations.
    full_frames:
        Number of frames used for promoted full evaluations.
    rung_fractions:
        Optional fractions of ``full_frames`` allocated to successive
        evaluation rungs. When provided, a simple ASHA-style scheduler routes
        configurations through these budgets and tracks per-rung counts,
        promotions and wall-clock time.
    state_path:
        Optional path where the optimiser state is checkpointed after each
        evaluation.
    run_index:
        Optional :class:`RunIndex` used to skip previously evaluated full
        configurations.
    When the optimiser is an instance of :class:`MCTS_H`, a unique
    ``mcts_run_id`` is generated and attached to every persisted run so
    evaluations can be traced back to the search session.
    """

    def __init__(
        self,
        base: Dict[str, float],
        gates: Iterable[int],
        fitness_fn: Callable[
            [Dict[str, float], Dict[str, float], Dict[str, float]], float
        ],
        optimizer: "Optimizer",
        seed: int = 0,
        *,
        proxy_frames: int = 300,
        full_frames: int = 3000,
        state_path: str | Path | None = None,
        run_index: RunIndex | None = None,
        rung_fractions: Sequence[float] | None = None,
    ) -> None:
        self.base = dict(base)
        self.gates = list(gates)
        self.fitness_fn = fitness_fn
        self.optimizer = optimizer
        self.seed = seed
        self._normalizer = Normalizer()
        self._index = run_index or RunIndex()
        self._proxy_seen: set[Tuple[Tuple[str, float], ...]] = set()
        self._top_k_path = Path("experiments/top_k.json")
        self._hof_path = Path("experiments/hall_of_fame.json")
        self._hof: List[Dict[str, Any]] = []
        self._state_path = Path(state_path) if state_path else None
        self.proxy_frames = int(proxy_frames)
        self.full_frames = int(full_frames)
        self._rung_frames: List[int] | None = (
            [int(full_frames * f) for f in rung_fractions] if rung_fractions else None
        )
        if self._rung_frames:
            self._pending: List[Tuple[Dict[str, float], int]] = []
            self._rung_data: List[List[Tuple[float, Dict[str, float]]]] = [
                [] for _ in self._rung_frames[:-1]
            ]
            self._rung_counts = [0 for _ in self._rung_frames]
            self._promotions = [0 for _ in self._rung_frames[:-1]]
            self._asha_eta = 2.0
            self._rung_times = [0.0 for _ in self._rung_frames]
        self._mcts_run_id: Optional[str] = None
        if isinstance(self.optimizer, MCTS_H):
            self._mcts_run_id = allocate_run_dir()[0]

    # ------------------------------------------------------------------
    def _checkpoint(self) -> None:
        if self._state_path and hasattr(self.optimizer, "save"):
            self.optimizer.save(self._state_path)

    # ------------------------------------------------------------------
    def run_next(self) -> Optional[OptimizerResult]:
        """Evaluate the next configuration from the optimizer."""

        if self.optimizer.done():
            if self._rung_frames:
                stats = self.rung_stats()
                logger.info(
                    "ASHA rungs: counts=%s promotions=%s times=%s",
                    stats["rung_counts"],
                    stats["promotion_fractions"],
                    stats.get("rung_times"),
                )
            return None
        if self._rung_frames:
            if self._pending:
                cfg, rung = self._pending.pop(0)
            else:
                cfg = self.optimizer.suggest(1)[0]
                rung = 0
            raw = self._normalizer.to_raw(self.base, cfg)
            raw.setdefault("seed", self.seed)
            frames = self._rung_frames[rung]
            t0 = time.perf_counter()
            metrics = run_gates(raw, self.gates, frames=frames)
            self._rung_times[rung] += time.perf_counter() - t0
            inv = checks.from_metrics(metrics)
            fit = self.fitness_fn(metrics, inv, cfg)
            self._rung_counts[rung] += 1
            if rung < len(self._rung_frames) - 1:
                self._rung_data[rung].append((float(fit), cfg))
                data = self._rung_data[rung]
                penal = float(fit)
                if len(data) >= self._asha_eta:
                    k = max(1, int(len(data) / self._asha_eta))
                    thresh = sorted(data, key=lambda x: x[0])[k - 1][0]
                    if float(fit) <= thresh:
                        self._pending.append((cfg, rung + 1))
                        self._promotions[rung] += 1
                    else:
                        penal = float(thresh)
                self.optimizer.observe([{"config": cfg, "fitness_proxy": penal}])
            else:
                res = {"config": cfg, "fitness": float(fit)}
                self.optimizer.observe([res])
                run_id, abs_path, rel_path = allocate_run_dir()
                key_hash = run_key(
                    {
                        "groups": cfg,
                        "toggles": {},
                        "seed": int(raw["seed"]),
                        "gates": self.gates,
                    }
                )
                manifest = {
                    "run_id": run_id,
                    "run_key": key_hash,
                    "groups": cfg,
                    "toggles": {},
                    "seed": int(raw["seed"]),
                    "gates": self.gates,
                }
                if self._mcts_run_id is not None:
                    manifest["mcts_run_id"] = self._mcts_run_id
                result_payload = {
                    "status": "ok",
                    "metrics": metrics,
                    "invariants": inv,
                    "fitness": float(fit),
                }
                persist_run(raw, result_payload, abs_path, manifest=manifest)
                self._index.mark(key_hash, rel_path)
                entry = TopKEntry(
                    run_id=run_id,
                    fitness=-float(fit),
                    objectives={"f0": float(fit)},
                    groups=cfg,
                    toggles={},
                    seed=int(raw["seed"]),
                    path=rel_path,
                )
                update_top_k([entry], self._top_k_path)
                self._hof.append(
                    {
                        "run_id": run_id,
                        "fitness": -float(fit),
                        "objectives": {"f0": float(fit)},
                        "path": rel_path,
                    }
                )
                save_hall_of_fame(self._hof, self._hof_path)
                self._checkpoint()
                return OptimizerResult(cfg, "full", float(fit), rel_path)
            self._checkpoint()
            return OptimizerResult(cfg, "proxy", float(fit))
        # legacy single proxy/full path
        cfg = self.optimizer.suggest(1)[0]
        key = tuple(sorted(cfg.items()))
        full = key in self._proxy_seen
        if not full and hasattr(self.optimizer, "_suggest_full"):
            if key in getattr(self.optimizer, "_suggest_full"):
                full = True
                getattr(self.optimizer, "_suggest_full").discard(key)

        raw = self._normalizer.to_raw(self.base, cfg)
        raw.setdefault("seed", self.seed)

        if not full:
            metrics = run_gates(raw, self.gates, frames=self.proxy_frames)
            inv = checks.from_metrics(metrics)
            fit = self.fitness_fn(metrics, inv, cfg)
            self._proxy_seen.add(key)
            self.optimizer.observe([{"config": cfg, "fitness_proxy": float(fit)}])
            self._checkpoint()
            return OptimizerResult(cfg, "proxy", float(fit))

        metrics = run_gates(raw, self.gates, frames=self.full_frames)
        inv = checks.from_metrics(metrics)
        fit = self.fitness_fn(metrics, inv, cfg)
        self._proxy_seen.discard(key)
        res = {"config": cfg, "fitness": float(fit)}
        self.optimizer.observe([res])
        self._checkpoint()

        run_id, abs_path, rel_path = allocate_run_dir()
        key_hash = run_key(
            {
                "groups": cfg,
                "toggles": {},
                "seed": int(raw["seed"]),
                "gates": self.gates,
            }
        )
        manifest = {
            "run_id": run_id,
            "run_key": key_hash,
            "groups": cfg,
            "toggles": {},
            "seed": int(raw["seed"]),
            "gates": self.gates,
        }
        if self._mcts_run_id is not None:
            manifest["mcts_run_id"] = self._mcts_run_id
        result_payload = {
            "status": "ok",
            "metrics": metrics,
            "invariants": inv,
            "fitness": float(fit),
        }
        persist_run(raw, result_payload, abs_path, manifest=manifest)
        self._index.mark(key_hash, rel_path)

        entry = TopKEntry(
            run_id=run_id,
            fitness=-float(fit),
            objectives={"f0": float(fit)},
            groups=cfg,
            toggles={},
            seed=int(raw["seed"]),
            path=rel_path,
        )
        update_top_k([entry], self._top_k_path)
        self._hof.append(
            {
                "run_id": run_id,
                "fitness": -float(fit),
                "objectives": {"f0": float(fit)},
                "path": rel_path,
            }
        )
        save_hall_of_fame(self._hof, self._hof_path)
        return OptimizerResult(cfg, "full", float(fit), rel_path)

    # ------------------------------------------------------------------
    def rung_stats(self) -> Dict[str, Any]:
        """Return evaluation counts, promotion fractions and times per rung."""

        if not self._rung_frames:
            return {}
        fracs = []
        for i, cnt in enumerate(self._rung_counts[:-1]):
            total = cnt or 1
            fracs.append(self._promotions[i] / total)
        stats = {
            "rung_counts": list(self._rung_counts),
            "promotion_fractions": fracs,
            "rung_times": list(self._rung_times),
        }
        logger.info(
            "ASHA rungs: counts=%s promotions=%s times=%s",
            stats["rung_counts"],
            stats["promotion_fractions"],
            stats["rung_times"],
        )
        return stats
