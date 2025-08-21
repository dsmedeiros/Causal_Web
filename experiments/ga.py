from __future__ import annotations

"""Genetic Algorithm utilities for experiment optimisation.

This module implements a minimal GA operating on dimensionless groups and
optional discrete engine toggles.  Fitness evaluation delegates to a user
provided callback which can reuse the same gate runner as the DOE queue
manager.
"""

from dataclasses import dataclass, field
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
import argparse
import importlib.util
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
from .index import RunIndex, run_key
from .artifacts import (
    TopKEntry,
    update_top_k,
    save_hall_of_fame,
    persist_run,
    allocate_run_dir,
)
from .runner import _load_base_config


def dominates(a: "Genome", b: "Genome") -> bool:
    """Return ``True`` when genome ``a`` Pareto-dominates ``b``."""

    if not isinstance(a.fitness, Sequence) or not isinstance(b.fitness, Sequence):
        return False
    a_better = False
    for ax, bx in zip(a.fitness, b.fitness):
        if ax > bx:
            return False
        if ax < bx:
            a_better = True
    return a_better


def fast_non_dominated_sort(pop: Sequence["Genome"]) -> List[List["Genome"]]:
    """Split ``pop`` into Pareto fronts using fast non-dominated sorting."""

    fronts: List[List[Genome]] = [[]]
    for p in pop:
        p.S = []
        p.n = 0
        for q in pop:
            if dominates(p, q):
                p.S.append(q)
            elif dominates(q, p):
                p.n += 1
        if p.n == 0:
            p.rank = 0
            fronts[0].append(p)
    i = 0
    while fronts[i]:
        next_front: List[Genome] = []
        for p in fronts[i]:
            for q in p.S:
                q.n -= 1
                if q.n == 0:
                    q.rank = i + 1
                    next_front.append(q)
        i += 1
        fronts.append(next_front)
    return fronts[:-1]


def crowding_distance(front: Sequence["Genome"]) -> None:
    """Compute crowding distance for ``front`` in-place."""

    if not front:
        return
    m = len(front[0].fitness) if isinstance(front[0].fitness, Sequence) else 0
    for p in front:
        p.cd = 0.0
    for k in range(m):
        front.sort(key=lambda g: g.fitness[k])
        front[0].cd = front[-1].cd = float("inf")
        minv, maxv = front[0].fitness[k], front[-1].fitness[k]
        denom = max(maxv - minv, 1e-12)
        for i in range(1, len(front) - 1):
            p = front[i]
            p.cd += (front[i + 1].fitness[k] - front[i - 1].fitness[k]) / denom


@dataclass
class Genome:
    """Container describing a single genome in the population.

    ``run_id`` and ``run_path`` are populated when the genome has been
    evaluated and its configuration/result persisted to disk. ``invariants``
    captures any constraint metrics returned during the last evaluation.
    """

    groups: Dict[str, float]
    toggles: Dict[str, int]
    fitness: Optional[Union[float, Sequence[float]]] = None
    invariants: Optional[Dict[str, float | bool]] = None
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
    run_index:
        Optional :class:`RunIndex` used to track completed runs and avoid
        duplicate evaluations.
    force:
        When ``True`` re-evaluate genomes even if present in the run index.
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
        run_index: RunIndex | None = None,
        force: bool = False,
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
        self._index = run_index or RunIndex()
        self._force = force
        self._archive: List[Genome] = []
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

        seed = self._seed_for_genome(genome)
        cfg = {
            "groups": genome.groups,
            "toggles": genome.toggles,
            "seed": seed,
            "gates": self.gates,
        }
        key = run_key(cfg)
        info = None if self._force else self._index.get(key)
        if info is not None:
            run_dir = self._index.runs_root / info
            try:
                res = json.loads((run_dir / "result.json").read_text())
            except Exception:
                res = {}
            genome.fitness = res.get("fitness")
            genome.invariants = res.get("invariants")
            try:
                manifest = json.loads((run_dir / "manifest.json").read_text())
                genome.run_id = manifest.get("run_id")
            except Exception:
                genome.run_id = None
            genome.run_path = str(Path("runs") / info)
            return genome.fitness

        raw = self._normalizer.to_raw(self.base, genome.groups)
        raw.update(genome.toggles)
        raw.setdefault("seed", seed)
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
        genome.invariants = res.get("invariants")
        if res.get("status") == "ok":
            rid, abs_path, rel_path = allocate_run_dir()
            manifest = {
                "run_id": rid,
                "run_key": key,
                "groups": genome.groups,
                "toggles": genome.toggles,
                "seed": seed,
                "gates": self.gates,
            }
            persist_run(raw, res, abs_path, manifest=manifest)
            self._index.mark(key, rel_path)
            genome.run_id = rid
            genome.run_path = rel_path
        return genome.fitness

    # ------------------------------------------------------------------
    def _tournament(self) -> Genome:
        """Return the best genome among ``tournament_k`` random samples."""

        samples = self.rng.sample(self.population, self.tournament_k)
        if any(hasattr(g, "rank") for g in samples):
            return min(
                samples,
                key=lambda g: (
                    getattr(g, "rank", float("inf")),
                    -getattr(g, "cd", 0.0),
                ),
            )
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
    def _update_archive(self, front: Sequence[Genome]) -> None:
        """Update the persistent Pareto archive with ``front``."""

        combined_map: Dict[
            Tuple[Tuple[Tuple[str, float], ...], Tuple[Tuple[str, int], ...]], Genome
        ] = {}
        for g in list(self._archive) + list(front):
            key = (
                tuple(sorted(g.groups.items())),
                tuple(sorted(g.toggles.items())),
            )
            if key not in combined_map:
                combined_map[key] = Genome(
                    dict(g.groups), dict(g.toggles), g.fitness, g.run_id, g.run_path
                )
        combined = list(combined_map.values())
        fronts = fast_non_dominated_sort(combined)
        archive = fronts[0] if fronts else []
        if len(archive) > self.population_size:
            crowding_distance(archive)
            archive.sort(key=lambda g: -g.cd)
            archive = archive[: self.population_size]
        self._archive = archive

    # ------------------------------------------------------------------
    def step(self) -> Genome:
        """Advance the population by one generation."""

        for g in self.population:
            if g.fitness is None:
                self._evaluate(g)
        multi = any(isinstance(g.fitness, Sequence) for g in self.population)
        if multi:
            fronts = fast_non_dominated_sort(self.population)
            for front in fronts:
                crowding_distance(front)
            self._update_archive(fronts[0])
            self.population.sort(key=lambda g: (g.rank, -g.cd))
        else:
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
        """Return genomes in the persistent Pareto archive."""

        if self._archive:
            return list(self._archive)
        fronts = fast_non_dominated_sort(
            [g for g in self.population if isinstance(g.fitness, Sequence)]
        )
        return fronts[0] if fronts else []

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
        hof_entries: List[Dict[str, Any]] = []
        # Persist Pareto archive for multi-objective runs; fall back to the
        # hall-of-fame in single-objective scenarios so callers always receive a
        # populated archive.
        source = self._archive or [g for _, g in self._hall_of_fame]
        for g in source:
            if g.run_id is None or g.run_path is None:
                continue
            obj = (
                {f"f{i}": float(v) for i, v in enumerate(g.fitness)}
                if isinstance(g.fitness, Sequence)
                else {}
            )
            hof_entries.append(
                {
                    "run_id": g.run_id,
                    "fitness": self._score(g),
                    "objectives": obj,
                    "path": g.run_path,
                }
            )
        update_top_k(top_entries, Path(top_k_path), k)
        save_hall_of_fame(hof_entries, Path(hall_of_fame_path))

    # ------------------------------------------------------------------
    def _write_config(self, genome: Genome, path: str) -> None:
        """Persist ``genome`` configuration to ``path`` as YAML."""

        cfg: Dict[str, Any] | None = None
        if genome.run_path:
            try:
                cfg = json.loads(
                    (Path("experiments") / genome.run_path / "config.json").read_text()
                )
            except Exception:  # pragma: no cover - fall back to materialised config
                cfg = None
        if cfg is None:
            raw = self._normalizer.to_raw(self.base, genome.groups)
            raw.update(genome.toggles)
            cfg = {
                k: float(v) if isinstance(v, np.floating) else v for k, v in raw.items()
            }
        import yaml

        with open(path, "w", encoding="utf8") as fh:
            yaml.safe_dump(cfg, fh)

    # ------------------------------------------------------------------
    def promote_best(self, path: str) -> None:
        """Write the best genome's configuration to ``path`` as YAML."""

        best = max(self.population, key=self._score)
        self._write_config(best, path)

    # ------------------------------------------------------------------
    def promote_pareto(self, index: int, path: str) -> None:
        """Persist the ``index``-th Pareto genome to ``path``."""

        front = self.pareto_front()
        if not 0 <= index < len(front):
            raise IndexError("pareto index out of range")
        self._write_config(front[index], path)

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


@dataclass
class GAConfig:
    """Configuration for command-line GA runs."""

    group_ranges: Dict[str, Tuple[float, float]]
    toggles: Dict[str, Sequence[int]] = field(default_factory=dict)
    gates: List[int] = field(default_factory=list)
    population_size: int = 10
    generations: int = 1
    elite: int = 1
    mutation_sigma: float = 0.1
    tournament_k: int = 2
    seed: int = 0

    @classmethod
    def from_mapping(cls, data: Dict[str, Any]) -> "GAConfig":
        required = {"groups"}
        missing = required - data.keys()
        if missing:
            raise KeyError(
                f"GA configuration missing keys: {', '.join(sorted(missing))}"
            )
        group_ranges = {
            k: (float(v[0]), float(v[1])) for k, v in data["groups"].items()
        }
        toggles = {k: [int(x) for x in v] for k, v in data.get("toggles", {}).items()}
        return cls(
            group_ranges,
            toggles,
            list(data.get("gates", [])),
            int(data.get("population_size", 10)),
            int(data.get("generations", 1)),
            int(data.get("elite", 1)),
            float(data.get("mutation_sigma", 0.1)),
            int(data.get("tournament_k", 2)),
            int(data.get("seed", 0)),
        )


def _load_ga_config(path: Path) -> GAConfig:
    """Return a :class:`GAConfig` loaded from ``path``."""

    if path.suffix in {".yaml", ".yml"}:
        import yaml

        data = yaml.safe_load(path.read_text())
    elif path.suffix == ".toml":
        import tomllib

        data = tomllib.loads(path.read_text())
    else:
        raise ValueError(f"Unsupported config extension: {path.suffix}")
    return GAConfig.from_mapping(data)


def main(argv: Iterable[str] | None = None) -> None:
    """Run a GA optimisation from configuration files."""

    parser = argparse.ArgumentParser(description="Run GA experiments")
    parser.add_argument("--exp", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--fitness", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-evaluate genomes even if present in the run index",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    cfg = _load_ga_config(args.exp)
    base = _load_base_config(args.base)
    spec = importlib.util.spec_from_file_location("ga_fitness", args.fitness)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    fitness_fn = getattr(mod, "fitness")

    ga = GeneticAlgorithm(
        base,
        cfg.group_ranges,
        cfg.toggles,
        cfg.gates,
        fitness_fn,
        population_size=cfg.population_size,
        elite=cfg.elite,
        mutation_sigma=cfg.mutation_sigma,
        tournament_k=cfg.tournament_k,
        seed=cfg.seed,
        run_index=RunIndex(),
        force=args.force,
    )
    ga.run(cfg.generations)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    ga.promote_best(args.out)


if __name__ == "__main__":  # pragma: no cover
    main()
