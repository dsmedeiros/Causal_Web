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
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np

from config.normalizer import Normalizer
from invariants import checks
from .gates import run_gates
from .runner import _latin_hypercube


@dataclass
class RunStatus:
    """Track the state of a single experiment run.

    ``run_id`` and ``path`` are populated once a run has been persisted to
    disk.  ``path`` is stored relative to the ``experiments`` directory so it
    can be embedded directly into Top-K artifacts.
    """

    state: str = "queued"
    invariants: Optional[Dict[str, float]] = None
    fitness: Optional[float] = None
    error: Optional[str] = None
    run_id: Optional[str] = None
    path: Optional[str] = None


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
    ) -> None:
        self.base = dict(base)
        self.gates = list(gates)
        self.fitness_fn = fitness_fn
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self._runs: List[Tuple[Dict[str, float], RunStatus]] = []
        self._normalizer = Normalizer()
        self._client = client

    # ------------------------------------------------------------------
    # queue construction
    def enqueue_lhs(self, groups: Dict[str, Tuple[float, float]], samples: int) -> None:
        """Enqueue samples via Latin Hypercube sampling.

        Parameters
        ----------
        groups:
            Mapping of group names to ``(low, high)`` ranges.
        samples:
            Number of samples to generate.
        """

        names = list(groups.keys())
        ranges = np.array([groups[n] for n in names], dtype=float)
        unit = _latin_hypercube(samples, len(names), self._rng)
        lows, highs = ranges[:, 0], ranges[:, 1]
        scaled = lows[None, :] + unit * (highs - lows)[None, :]
        for i in range(samples):
            cfg = dict(zip(names, scaled[i]))
            self._runs.append((cfg, RunStatus()))

    def enqueue_grid(
        self, groups: Dict[str, Tuple[float, float]], steps: Dict[str, int]
    ) -> None:
        """Enqueue a uniform grid sweep.

        Parameters
        ----------
        groups:
            Mapping of group names to ``(low, high)`` ranges.
        steps:
            Number of steps along each group dimension.
        """

        names = list(groups.keys())
        axes: List[np.ndarray] = []
        for n in names:
            low, high = groups[n]
            cnt = steps.get(n, 1)
            axes.append(np.linspace(low, high, cnt))
        for combo in itertools.product(*axes):
            cfg = dict(zip(names, combo))
            self._runs.append((cfg, RunStatus()))

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
                status.state = "running"
                try:
                    raw = self._normalizer.to_raw(self.base, groups)
                    metrics = run_gates(raw, self.gates)
                    inv = checks.from_metrics(metrics)
                    status.invariants = inv
                    if self.fitness_fn is not None:
                        status.fitness = self.fitness_fn(metrics, inv, groups, {})
                    status.state = "finished"
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
                status.state = "running"
                raw = self._normalizer.to_raw(self.base, groups)
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
