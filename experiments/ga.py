from __future__ import annotations

"""Genetic Algorithm utilities for experiment optimisation.

This module implements a minimal GA operating on dimensionless groups and
optional discrete engine toggles.  Fitness evaluation delegates to a user
provided callback which can reuse the same gate runner as the DOE queue
manager.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)
import asyncio
import random
import json
import hashlib
import pickle
import time
from concurrent.futures import TimeoutError
import numpy as np

from config.normalizer import Normalizer
from invariants import checks
from .gates import run_gates
from .artifacts import (
    TopKEntry,
    update_top_k,
    save_hall_of_fame,
    persist_run,
    allocate_run_dir,
)


@dataclass
class Genome:
    """Container describing a single genome in the population.

    ``run_id`` and ``run_path`` are populated when the genome has been
    evaluated and its configuration/result persisted to disk.
    """

    groups: Dict[str, float]
    toggles: Dict[str, int]
    fitness: Optional[Union[float, Sequence[float]]] = None
    run_id: Optional[str] = None
    run_path: Optional[str] = None


class GeneticAlgorithm:
    """Evolve configurations over dimensionless groups and engine toggles.

    Parameters
    ----------
    base:
        Baseline raw configuration used to materialise groups.
    group_ranges:
        Mapping of group names to ``(low, high)`` ranges.
    toggle_choices:
        Mapping of toggle names to the discrete set of integer choices.
    gates:
        Sequence of gate identifiers executed during fitness evaluation.
    fitness_fn:
        Callback returning a scalar or vector fitness given metrics, invariants,
        groups and toggles.
    population_size:
        Number of genomes in the population.
    elite:
        Number of top genomes carried over unchanged each generation.
    mutation_sigma:
        Standard deviation of Gaussian noise applied during mutation in the
        units of the respective group range.
    tournament_k:
        Number of genomes sampled during tournament selection.
    seed:
        RNG seed controlling initial population and stochastic operators.
    client:
        Optional IPC client used to delegate fitness evaluation to a running
        engine.
    loop:
        AsyncIO event loop associated with ``client``.  Required when
        ``client`` is provided so evaluation requests can be scheduled and
        results returned via ``ExperimentStatus`` messages.
    """

    def __init__(
        self,
        base: Dict[str, float],
        group_ranges: Dict[str, Tuple[float, float]],
        toggle_choices: Dict[str, Sequence[int]],
        gates: Iterable[int],
        fitness_fn: Callable[
            [Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, int]],
            Union[float, Sequence[float]],
        ],
        population_size: int = 10,
        elite: int = 1,
        mutation_sigma: float = 0.1,
        tournament_k: int = 2,
        seed: int = 0,
        client: Any | None = None,
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        self.base = dict(base)
        self.group_ranges = dict(group_ranges)
        self.toggle_choices = {k: list(v) for k, v in toggle_choices.items()}
        self.gates = list(gates)
        self.fitness_fn = fitness_fn
        self.population_size = population_size
        self.elite = elite
        self.mutation_sigma = mutation_sigma
        self.tournament_k = tournament_k
        self.rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)
        self._normalizer = Normalizer()
        self.population: List[Genome] = []
        self.history: List[float] = []
        self._hall_of_fame: List[Tuple[int, Genome]] = []
        self._generation = 0
        self._client = client
        self._loop = loop
        self._pending: Dict[int, Tuple[Genome, asyncio.Future, int]] = {}
        self._next_id = 0
        self._init_population()

    # ------------------------------------------------------------------
    def set_client(self, client: Any | None) -> None:
        """Assign an IPC ``client`` for engine communication."""

        self._client = client

    # ------------------------------------------------------------------
    def set_event_loop(self, loop: asyncio.AbstractEventLoop | None) -> None:
        """Assign the AsyncIO ``loop`` used for IPC callbacks."""

        self._loop = loop

    # ------------------------------------------------------------------
    def _seed_for_genome(self, genome: Genome) -> int:
        """Return a deterministic seed for ``genome`` based on its genes."""

        payload = json.dumps(
            [sorted(genome.groups.items()), sorted(genome.toggles.items())],
            sort_keys=True,
        )
        digest = hashlib.sha256(payload.encode("utf8")).hexdigest()
        return int(digest, 16) & 0xFFFFFFFF

    # ------------------------------------------------------------------
    async def _async_eval_genome(self, genome: Genome, seed: int) -> Dict[str, Any]:
        """Schedule ``genome`` evaluation on the engine loop."""

        rid = self._next_id
        self._next_id += 1
        fut: asyncio.Future = self._loop.create_future()
        self._pending[rid] = (genome, fut, seed)
        raw = self._normalizer.to_raw(self.base, genome.groups)
        raw.update(genome.toggles)
        raw.setdefault("seed", seed)
        await self._client.send(
            {
                "ExperimentControl": {
                    "action": "run",
                    "id": rid,
                    "config": raw,
                    "gates": self.gates,
                }
            }
        )
        return await fut

    # ------------------------------------------------------------------
    def evaluate_blocking(
        self, genome: Genome, timeout_s: float = 60.0
    ) -> Dict[str, Any]:
        """Evaluate ``genome`` synchronously with retry and timeout handling."""

        seed = self._seed_for_genome(genome)
        for attempt in range(2):
            fut = asyncio.run_coroutine_threadsafe(
                self._async_eval_genome(genome, seed), self._loop
            )
            try:
                res: Dict[str, Any] = fut.result(timeout=timeout_s * (2**attempt))
            except TimeoutError:
                res = {"status": "timeout", "message": "evaluation timed out"}
            if res.get("status") == "ok":
                return res
            if attempt == 0:
                self._loop.run_until_complete(asyncio.sleep(2**attempt))
        return res

    # ------------------------------------------------------------------
    def _init_population(self) -> None:
        """Initialise the population with random genomes."""

        names = list(self.group_ranges.keys())
        lows = np.array([self.group_ranges[n][0] for n in names], dtype=float)
        highs = np.array([self.group_ranges[n][1] for n in names], dtype=float)
        unit = self._np_rng.random((self.population_size, len(names)))
        groups_arr = lows[None, :] + unit * (highs - lows)[None, :]
        for i in range(self.population_size):
            groups = {n: float(groups_arr[i, j]) for j, n in enumerate(names)}
            toggles = {
                k: self.rng.choice(v) if v else 0
                for k, v in self.toggle_choices.items()
            }
            self.population.append(Genome(groups, toggles))

    # ------------------------------------------------------------------
    def handle_status(self, msg: Dict[str, Any]) -> None:
        """Handle an ``ExperimentStatus`` message from the engine."""

        rid = msg.get("id")
        if rid is None:
            return
        pending = self._pending.get(rid)
        if pending is None:
            return
        genome, fut, _ = pending
        state = msg.get("state")
        if state == "failed":
            res = {
                "status": "engine_error",
                "message": msg.get("error", "run failed"),
            }
            if not fut.done():
                self._loop.call_soon_threadsafe(fut.set_result, res)
            self._pending.pop(rid, None)
            return
        metrics = msg.get("metrics", {})
        inv = msg.get("invariants", {})
        inv_ok = (
            inv.get("inv_causality_ok", True)
            and inv.get("inv_ancestry_ok", True)
            and abs(inv.get("inv_conservation_residual", 0.0)) <= 1e-9
            and abs(inv.get("inv_no_signaling_delta", 0.0)) <= 1e-9
        )
        if inv_ok:
            fit = self.fitness_fn(metrics, inv, genome.groups, genome.toggles)
            res = {
                "status": "ok",
                "fitness": fit,
                "metrics": metrics,
                "invariants": inv,
            }
        else:
            res = {
                "status": "invalid",
                "message": "invariant failure",
                "metrics": metrics,
                "invariants": inv,
            }
        if not fut.done():
            self._loop.call_soon_threadsafe(fut.set_result, res)
        self._pending.pop(rid, None)

    def _evaluate(self, genome: Genome) -> Union[float, Sequence[float]]:
        """Evaluate ``genome`` and persist its config/result on success."""

        raw = self._normalizer.to_raw(self.base, genome.groups)
        raw.update(genome.toggles)
        if self._client is None or self._loop is None:
            metrics = run_gates(raw, self.gates)
            inv = checks.from_metrics(metrics)
            inv_ok = (
                inv.get("inv_causality_ok", True)
                and inv.get("inv_ancestry_ok", True)
                and abs(inv.get("inv_conservation_residual", 0.0)) <= 1e-9
                and abs(inv.get("inv_no_signaling_delta", 0.0)) <= 1e-9
            )
            if inv_ok:
                fit = self.fitness_fn(metrics, inv, genome.groups, genome.toggles)
                res = {
                    "status": "ok",
                    "fitness": fit,
                    "metrics": metrics,
                    "invariants": inv,
                }
            else:
                res = {
                    "status": "invalid",
                    "metrics": metrics,
                    "invariants": inv,
                }
        else:
            res = self.evaluate_blocking(genome)
        genome.fitness = res.get("fitness") if res.get("status") == "ok" else None
        if res.get("status") == "ok":
            rid, abs_path, rel_path = allocate_run_dir()
            persist_run(raw, res, abs_path)
            genome.run_id = rid
            genome.run_path = rel_path
        return genome.fitness

    # ------------------------------------------------------------------
    def _tournament(self) -> Genome:
        """Return the best genome among ``tournament_k`` random samples."""

        samples = self.rng.sample(self.population, self.tournament_k)
        return max(samples, key=lambda g: self._score(g))

    # ------------------------------------------------------------------
    def _crossover(self, a: Genome, b: Genome) -> Genome:
        """Uniformly mix genes from parents ``a`` and ``b``."""

        groups = {
            k: (a.groups[k] if self.rng.random() < 0.5 else b.groups[k])
            for k in self.group_ranges.keys()
        }
        toggles = {
            k: (a.toggles[k] if self.rng.random() < 0.5 else b.toggles[k])
            for k in self.toggle_choices.keys()
        }
        return Genome(groups, toggles)

    # ------------------------------------------------------------------
    def _mutate(self, genome: Genome) -> None:
        """Apply Gaussian noise to groups and random flips to toggles."""

        for k, (low, high) in self.group_ranges.items():
            span = high - low
            genome.groups[k] += self._np_rng.normal(0.0, self.mutation_sigma * span)
            genome.groups[k] = float(np.clip(genome.groups[k], low, high))
        for k, choices in self.toggle_choices.items():
            if self.rng.random() < 0.1 and choices:
                genome.toggles[k] = self.rng.choice(choices)

    # ------------------------------------------------------------------
    def step(self) -> Genome:
        """Advance the population by one generation."""

        for g in self.population:
            if g.fitness is None:
                self._evaluate(g)
        self.population.sort(key=lambda g: self._score(g), reverse=True)
        best = self.population[0]
        self.history.append(self._score(best))
        self._hall_of_fame.append((self._generation, best))

        next_pop: List[Genome] = self.population[: self.elite]
        while len(next_pop) < self.population_size:
            p1 = self._tournament()
            p2 = self._tournament()
            child = self._crossover(p1, p2)
            self._mutate(child)
            next_pop.append(child)
        self.population = next_pop
        for g in self.population[self.elite :]:
            g.fitness = None
        self._generation += 1
        return best

    # ------------------------------------------------------------------
    def _score(self, genome: Genome) -> float:
        """Return a scalar fitness score for ``genome``."""

        fit = genome.fitness
        if fit is None:
            return -np.inf
        if isinstance(fit, Sequence):
            return float(fit[0])
        return float(fit)

    # ------------------------------------------------------------------
    def pareto_front(self) -> List[Genome]:
        """Return non-dominated genomes for multi-objective fitness."""

        seq = [g for g in self.population if isinstance(g.fitness, Sequence)]
        front: List[Genome] = []
        for g in seq:
            dominated = False
            for h in seq:
                if g is h:
                    continue
                if all(hf >= gf for hf, gf in zip(h.fitness, g.fitness)) and any(
                    hf > gf for hf, gf in zip(h.fitness, g.fitness)
                ):
                    dominated = True
                    break
            if not dominated:
                front.append(g)
        return front

    # ------------------------------------------------------------------
    def run(self, generations: int) -> Genome:
        """Execute ``generations`` evolution steps and return the best genome."""

        best = None
        for _ in range(generations):
            best = self.step()
        return best if best is not None else self.population[0]

    # ------------------------------------------------------------------
    def save_artifacts(
        self,
        top_k_path: str,
        hall_of_fame_path: str,
        k: int = 50,
    ) -> None:
        """Persist top-K and hall-of-fame summaries to disk."""

        top_entries: List[TopKEntry] = []
        hof_entries: List[Dict[str, Any]] = []
        for gen, g in self._hall_of_fame:
            if g.run_id is None or g.run_path is None:
                continue
            obj = (
                {f"f{i}": float(v) for i, v in enumerate(g.fitness)}
                if isinstance(g.fitness, Sequence)
                else {}
            )
            top_entries.append(
                TopKEntry(
                    run_id=g.run_id,
                    fitness=self._score(g),
                    objectives=obj,
                    groups=g.groups,
                    toggles=g.toggles,
                    seed=self._seed_for_genome(g),
                    path=g.run_path,
                )
            )
            hof_entries.append(
                {
                    "gen": gen,
                    "run_id": g.run_id,
                    "fitness": self._score(g),
                    "objectives": obj,
                    "path": g.run_path,
                }
            )
        update_top_k(top_entries, Path(top_k_path), k)
        save_hall_of_fame(hof_entries, Path(hall_of_fame_path))

    # ------------------------------------------------------------------
    def promote_best(self, path: str) -> None:
        """Write the best genome's configuration to ``path`` as YAML.

        If the genome has a persisted run directory, the configuration is
        loaded from the saved ``config.json``.  Otherwise it is materialised
        from the genome's groups and toggles.
        """

        best = max(self.population, key=self._score)
        cfg: Dict[str, Any] | None = None
        if best.run_path:
            try:
                cfg = json.loads(
                    (Path("experiments") / best.run_path / "config.json").read_text()
                )
            except Exception:  # pragma: no cover - fall back to materialised config
                cfg = None
        if cfg is None:
            raw = self._normalizer.to_raw(self.base, best.groups)
            raw.update(best.toggles)
            cfg = {
                k: float(v) if isinstance(v, np.floating) else v for k, v in raw.items()
            }
        import yaml

        with open(path, "w", encoding="utf8") as fh:
            yaml.safe_dump(cfg, fh)

    # ------------------------------------------------------------------
    def save_checkpoint(self, path: str) -> None:
        """Persist the GA state to ``path`` for later restoration.

        Pending evaluations are recorded with their deterministic seeds so they
        can be resubmitted when the checkpoint is loaded.
        """

        data = {
            "base": self.base,
            "group_ranges": self.group_ranges,
            "toggle_choices": self.toggle_choices,
            "gates": self.gates,
            "population": [
                {
                    "groups": g.groups,
                    "toggles": g.toggles,
                    "fitness": g.fitness,
                }
                for g in self.population
            ],
            "history": self.history,
            "rng_state": self.rng.getstate(),
            "np_rng_state": self._np_rng.bit_generator.state,
            "next_id": self._next_id,
            "pending": [
                {
                    "groups": g.groups,
                    "toggles": g.toggles,
                    "seed": seed,
                }
                for _, (g, _, seed) in self._pending.items()
            ],
        }
        with open(path, "wb") as fh:
            pickle.dump(data, fh)

    # ------------------------------------------------------------------
    @classmethod
    def load_checkpoint(
        cls,
        path: str,
        fitness_fn: Callable[
            [Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, int]],
            Union[float, Sequence[float]],
        ],
        client: Any | None = None,
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> "GeneticAlgorithm":
        """Return a :class:`GeneticAlgorithm` restored from ``path``.

        Any in-flight evaluations captured in the checkpoint are automatically
        rescheduled if both ``client`` and ``loop`` are provided.
        """

        with open(path, "rb") as fh:
            data = pickle.load(fh)
        ga = cls(
            data["base"],
            data["group_ranges"],
            data["toggle_choices"],
            data["gates"],
            fitness_fn,
            population_size=len(data["population"]),
            seed=0,
            client=client,
            loop=loop,
        )
        ga.population = [
            Genome(d["groups"], d["toggles"], d.get("fitness"))
            for d in data["population"]
        ]
        ga.history = list(data["history"])
        ga.rng.setstate(data["rng_state"])
        ga._np_rng = np.random.default_rng()
        ga._np_rng.bit_generator.state = data["np_rng_state"]
        ga._next_id = data["next_id"]

        for item in data.get("pending", []):
            genome = Genome(item["groups"], item["toggles"])
            if client is not None and loop is not None:
                asyncio.run_coroutine_threadsafe(
                    ga._async_eval_genome(genome, item["seed"]), loop
                )

        return ga
