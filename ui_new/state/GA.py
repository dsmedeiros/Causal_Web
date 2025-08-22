from __future__ import annotations

"""Expose a simple Genetic Algorithm to QML panels."""

from typing import Dict, List, Optional
from pathlib import Path
import asyncio

from PySide6.QtCore import QFileSystemWatcher, QObject, Property, Signal, Slot

from experiments import GeneticAlgorithm
from experiments.artifacts import load_hall_of_fame, write_best_config
from experiments.fitness import VECTOR_FITNESS_LABELS
from ..ipc import Client


class GAModel(QObject):
    """Run a GA and expose population and fitness history."""

    populationChanged = Signal()
    historyChanged = Signal()
    paretoChanged = Signal()
    hallOfFameChanged = Signal()
    runningChanged = Signal()
    baselinePromoted = Signal(str)
    objectiveCountChanged = Signal()
    objectiveNamesChanged = Signal()
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
        self._group_ranges = {"Delta_over_W0": (0.0, 1.0)}
        self._toggles: Dict[str, List[int]] = {}

        def _fitness_multi(metrics, invariants, groups, toggles):
            g = groups["Delta_over_W0"]
            return (-abs(g - 0.5), -abs(g - 0.8))

        def _fitness_single(metrics, invariants, groups, toggles):
            return _fitness_multi(metrics, invariants, groups, toggles)[0]

        self._fitness_multi = _fitness_multi
        self._fitness_single = _fitness_single
        self._fitness = _fitness_single
        self._client: Optional[Client] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._population_size = 8
        self._mutation_rate = 0.1
        self._crossover_rate = 0.5
        self._elitism = 1
        self._max_generations = 10
        self._multi_objective = False
        self._ga = GeneticAlgorithm(
            self._base,
            self._group_ranges,
            self._toggles,
            [],
            self._fitness,
            population_size=self._population_size,
            elite=self._elitism,
            mutation_sigma=self._mutation_rate,
            seed=42,
        )
        self._population: List[dict] = []
        self._history: List[float] = []
        self._pareto: List[dict] = []
        self._obj_count = 0
        self._obj_names: List[str] = []
        self._hof_path = Path("experiments/hall_of_fame.json")
        data = load_hall_of_fame(self._hof_path)
        self._hof: List[dict] = list(data.get("archive", []))
        self._hof_watcher = QFileSystemWatcher([str(self._hof_path)])
        self._hof_watcher.fileChanged.connect(self._reload_hof)
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._generation = 0
        self._node_count = 0
        self._frontier = 0
        self._expansion_rate = 0.0
        self._promotion_rate = 0.0

    # ------------------------------------------------------------------
    def _reload_hof(self) -> None:
        data = load_hall_of_fame(self._hof_path)
        self._hof = list(data.get("archive", []))
        self.hallOfFameChanged.emit()
        if str(self._hof_path) not in self._hof_watcher.files():
            self._hof_watcher.addPath(str(self._hof_path))

    # ------------------------------------------------------------------
    @Slot()
    def step(self) -> None:
        """Advance the GA by one generation."""

        self._ga.step()
        self._population = []
        for g in self._ga.population:
            if isinstance(g.fitness, (list, tuple)):
                f0 = g.fitness[0]
                f1 = g.fitness[1] if len(g.fitness) > 1 else 0.0
            else:
                f0 = g.fitness or 0.0
                f1 = 0.0
            inv = getattr(g, "invariants", {}) or {}
            caus = bool(inv.get("inv_causality_ok", True))
            ances = bool(inv.get("inv_ancestry_ok", True))
            resid = abs(float(inv.get("inv_conservation_residual", 0.0))) <= 1e-9
            ns = abs(float(inv.get("inv_no_signaling_delta", 0.0))) <= 1e-9
            self._population.append(
                {
                    "fitness": f0,
                    "obj0": f0,
                    "obj1": f1,
                    "invCausality": caus,
                    "invAncestry": ances,
                    "invResidual": resid,
                    "invNoSignal": ns,
                    "path": g.run_path,
                    **g.groups,
                    **g.toggles,
                }
            )
        self._history = list(self._ga.history)
        pf: List[dict] = []
        front = self._ga.pareto_front()
        if front:
            self._obj_count = (
                len(front[0].fitness)
                if isinstance(front[0].fitness, (list, tuple))
                else 0
            )
            self._obj_names = VECTOR_FITNESS_LABELS[: self._obj_count]
        else:
            self._obj_count = 0
            self._obj_names = []
        for g in front:
            if isinstance(g.fitness, (list, tuple)):
                pf.append(
                    {
                        "rank": getattr(g, "rank", 0),
                        "crowding": getattr(g, "cd", 0.0),
                        "objs": list(g.fitness),
                        "path": g.run_path,
                    }
                )
        self._pareto = pf
        self._ga.save_artifacts(
            "experiments/top_k.json", "experiments/hall_of_fame.json"
        )
        data = load_hall_of_fame(Path("experiments/hall_of_fame.json"))
        self._hof = list(data.get("archive", []))
        self._generation += 1
        self.populationChanged.emit()
        self.historyChanged.emit()
        self.paretoChanged.emit()
        self.hallOfFameChanged.emit()
        self.objectiveCountChanged.emit()
        self.objectiveNamesChanged.emit()
        self._node_count = self._generation * self._population_size
        self._frontier = max(
            0, (self._max_generations - self._generation) * self._population_size
        )
        self._expansion_rate = (
            self._generation / self._max_generations if self._max_generations else 0.0
        )
        self._promotion_rate = (
            len(self._hof) / self._node_count if self._node_count else 0.0
        )
        self.statsChanged.emit()

    # ------------------------------------------------------------------
    @Slot()
    def exportBest(self) -> None:
        """Write the best genome to ``best_config.yaml``."""
        path = "experiments/best_config.yaml"
        self._ga.promote_best(path)
        self.baselinePromoted.emit(path)

    @Slot()
    def promoteBaseline(self) -> None:
        """Promote the best genome to the GA baseline."""

        best = max(
            self._ga.population,
            key=lambda g: (
                g.fitness if isinstance(g.fitness, (int, float)) else g.fitness[0]
            ),
        )
        raw = self._ga._normalizer.to_raw(self._ga.base, best.groups)
        self._ga.base.update(raw)
        path = "experiments/best_config.yaml"
        self._ga.promote_best(path)
        self.baselinePromoted.emit(path)

    @Slot(int)
    def promoteIndex(self, idx: int) -> None:  # noqa: N802 (Qt slot naming)
        """Promote the ``idx``-th Pareto genome."""
        path = "experiments/best_config.yaml"
        self._ga.promote_pareto(idx, path)
        self.baselinePromoted.emit(path)

    @Slot("QVariant")
    def promote(self, row: dict) -> None:
        """Persist ``row`` to ``best_config.yaml``."""
        if isinstance(row, dict):
            path = write_best_config(row)
            self.baselinePromoted.emit(path)

    def set_client(self, client: Client, loop: asyncio.AbstractEventLoop) -> None:
        """Attach a WebSocket ``client`` and event ``loop`` for engine integration."""

        self._client = client
        self._loop = loop
        self._ga.set_client(client)
        self._ga.set_event_loop(loop)

    def handle_status(self, msg: Dict) -> None:
        """Forward ``ExperimentStatus`` messages to the GA."""

        self._ga.handle_status(msg)

    # ------------------------------------------------------------------
    @Slot()
    def start(self) -> None:
        """Start evolving genomes until ``maxGenerations`` or paused."""

        if self._running:
            return
        fitness_fn = (
            self._fitness_multi if self._multi_objective else self._fitness_single
        )
        self._ga = GeneticAlgorithm(
            self._base,
            self._group_ranges,
            self._toggles,
            [],
            fitness_fn,
            population_size=self._population_size,
            elite=self._elitism,
            mutation_sigma=self._mutation_rate,
            seed=42,
        )
        self._population = []
        self._history = []
        self._pareto = []
        self._obj_count = 0
        self._obj_names = []
        self._generation = 0
        self._node_count = 0
        self._frontier = self._population_size * self._max_generations
        self._expansion_rate = 0.0
        self._promotion_rate = 0.0
        self._running = True
        self.runningChanged.emit()
        self.statsChanged.emit()

        async def _run() -> None:
            while self._running and self._generation < self._max_generations:
                self.step()
                await asyncio.sleep(0)
            self._running = False
            self.runningChanged.emit()

        self._task = asyncio.create_task(_run())

    # ------------------------------------------------------------------
    @Slot()
    def pause(self) -> None:
        """Pause the running GA."""

        self._running = False
        self.runningChanged.emit()

    # ------------------------------------------------------------------
    @Slot()
    def resume(self) -> None:
        """Resume evolution if previously paused."""

        if self._running or self._generation >= self._max_generations:
            return
        self.start()

    # ------------------------------------------------------------------
    def _get_population(self) -> List[dict]:
        return self._population

    population = Property("QVariant", _get_population, notify=populationChanged)

    def _get_history(self) -> List[float]:
        return self._history

    history = Property("QVariant", _get_history, notify=historyChanged)

    def _get_pareto(self) -> List[dict]:
        return self._pareto

    pareto = Property("QVariant", _get_pareto, notify=paretoChanged)

    def _get_obj_count(self) -> int:
        return self._obj_count

    objectiveCount = Property(int, _get_obj_count, notify=objectiveCountChanged)

    def _get_obj_names(self) -> List[str]:
        return self._obj_names

    objectiveNames = Property("QVariant", _get_obj_names, notify=objectiveNamesChanged)

    def _get_hof(self) -> List[dict]:
        return self._hof

    hallOfFame = Property("QVariant", _get_hof, notify=hallOfFameChanged)

    def _get_population_size(self) -> int:
        return self._population_size

    def _set_population_size(self, val: int) -> None:
        self._population_size = int(val)

    populationSize = Property(int, _get_population_size, _set_population_size)

    def _get_mutation_rate(self) -> float:
        return self._mutation_rate

    def _set_mutation_rate(self, val: float) -> None:
        self._mutation_rate = float(val)

    mutationRate = Property(float, _get_mutation_rate, _set_mutation_rate)

    def _get_crossover_rate(self) -> float:
        return self._crossover_rate

    def _set_crossover_rate(self, val: float) -> None:
        self._crossover_rate = float(val)

    crossoverRate = Property(float, _get_crossover_rate, _set_crossover_rate)

    def _get_elitism(self) -> int:
        return self._elitism

    def _set_elitism(self, val: int) -> None:
        self._elitism = int(val)

    elitism = Property(int, _get_elitism, _set_elitism)

    def _get_max_generations(self) -> int:
        return self._max_generations

    def _set_max_generations(self, val: int) -> None:
        self._max_generations = int(val)

    maxGenerations = Property(int, _get_max_generations, _set_max_generations)

    def _get_multi_objective(self) -> bool:
        return self._multi_objective

    def _set_multi_objective(self, val: bool) -> None:
        self._multi_objective = bool(val)

    multiObjective = Property(bool, _get_multi_objective, _set_multi_objective)

    def _get_running(self) -> bool:
        return self._running

    running = Property(bool, _get_running, notify=runningChanged)

    def _get_node_count(self) -> int:
        return self._node_count

    def _get_frontier(self) -> int:
        return self._frontier

    def _get_expansion_rate(self) -> float:
        return self._expansion_rate

    def _get_promotion_rate(self) -> float:
        return self._promotion_rate

    nodeCount = Property(int, _get_node_count, notify=statsChanged)
    frontier = Property(int, _get_frontier, notify=statsChanged)
    expansionRate = Property(float, _get_expansion_rate, notify=statsChanged)
    promotionRate = Property(float, _get_promotion_rate, notify=statsChanged)
