from __future__ import annotations

"""Genetic Algorithm utilities for experiment optimisation.

This module implements a minimal GA operating on dimensionless groups and
optional discrete engine toggles.  Fitness evaluation delegates to a user
provided callback which can reuse the same gate runner as the DOE queue
manager.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union
import asyncio
import random
import numpy as np

from config.normalizer import Normalizer
from invariants import checks
from .gates import run_gates


@dataclass
class Genome:
    """Container describing a single genome in the population."""

    groups: Dict[str, float]
    toggles: Dict[str, int]
    fitness: Optional[Union[float, Sequence[float]]] = None


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
        self._client = client
        self._loop = loop
        self._pending: Dict[int, Tuple[Genome, asyncio.Future]] = {}
        self._next_id = 0
        self._init_population()

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
        genome, fut = pending
        state = msg.get("state")
        if state == "failed":
            err = RuntimeError(msg.get("error", "run failed"))
            if not fut.done():
                fut.set_exception(err)
            self._pending.pop(rid, None)
            return
        metrics = msg.get("metrics", {})
        inv = msg.get("invariants", {})
        genome.fitness = self.fitness_fn(metrics, inv, genome.groups, genome.toggles)
        if not fut.done():
            fut.set_result(genome.fitness)
        self._pending.pop(rid, None)

    def _evaluate(self, genome: Genome) -> Union[float, Sequence[float]]:
        """Evaluate ``genome`` and cache its fitness."""

        if self._client is None or self._loop is None:
            raw = self._normalizer.to_raw(self.base, genome.groups)
            raw.update(genome.toggles)
            metrics = run_gates(raw, self.gates)
            inv = checks.from_metrics(metrics)
            genome.fitness = self.fitness_fn(
                metrics, inv, genome.groups, genome.toggles
            )
            return genome.fitness
        rid = self._next_id
        self._next_id += 1
        fut: asyncio.Future = self._loop.create_future()
        self._pending[rid] = (genome, fut)
        raw = self._normalizer.to_raw(self.base, genome.groups)
        raw.update(genome.toggles)
        asyncio.run_coroutine_threadsafe(
            self._client.send(
                {
                    "ExperimentControl": {
                        "action": "run",
                        "id": rid,
                        "config": raw,
                        "gates": self.gates,
                    }
                }
            ),
            self._loop,
        )
        genome.fitness = fut.result()
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
    def promote_best(self, path: str) -> None:
        """Write the best genome's raw configuration to ``path`` as YAML."""

        best = max(self.population, key=self._score)
        raw = self._normalizer.to_raw(self.base, best.groups)
        raw.update(best.toggles)
        import yaml

        with open(path, "w", encoding="utf8") as fh:
            yaml.safe_dump(
                {
                    k: float(v) if isinstance(v, np.floating) else v
                    for k, v in raw.items()
                },
                fh,
            )
