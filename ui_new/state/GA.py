from __future__ import annotations

"""Expose a simple Genetic Algorithm to QML panels."""

from typing import Dict, List, Optional
from pathlib import Path
import asyncio

from PySide6.QtCore import QObject, Property, Signal, Slot

from experiments import GeneticAlgorithm
from experiments.artifacts import load_hall_of_fame
from ..ipc import Client


class GAModel(QObject):
    """Run a GA and expose population and fitness history."""

    populationChanged = Signal()
    historyChanged = Signal()
    paretoChanged = Signal()
    hallOfFameChanged = Signal()

    def __init__(self) -> None:
        super().__init__()
        base = {
            "W0": 1.0,
            "alpha_leak": 1.0,
            "lambda_decay": 1.0,
            "b": 1.0,
            "prob": 0.5,
        }
        group_ranges = {"Delta_over_W0": (0.0, 1.0)}
        toggles: Dict[str, List[int]] = {}

        def fitness(metrics, invariants, groups, toggles):
            g = groups["Delta_over_W0"]
            return (-abs(g - 0.5), -abs(g - 0.8))

        self._client: Optional[Client] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._ga = GeneticAlgorithm(
            base,
            group_ranges,
            toggles,
            [],
            fitness,
            population_size=8,
            seed=42,
        )
        self._population: List[dict] = []
        self._history: List[float] = []
        self._pareto: List[List[float]] = []
        data = load_hall_of_fame(Path("experiments/hall_of_fame.json"))
        self._hof: List[dict] = list(data.get("archive", []))

    # ------------------------------------------------------------------
    @Slot()
    def step(self) -> None:
        """Advance the GA by one generation."""

        self._ga.step()
        self._population = [
            {
                "fitness": (
                    g.fitness[0]
                    if isinstance(g.fitness, (list, tuple))
                    else g.fitness or 0.0
                ),
                **g.groups,
                **g.toggles,
            }
            for g in self._ga.population
        ]
        self._history = list(self._ga.history)
        pf = [
            list(g.fitness[:2])
            for g in self._ga.pareto_front()
            if isinstance(g.fitness, (list, tuple)) and len(g.fitness) >= 2
        ]
        self._pareto = pf
        self._ga.save_artifacts(
            "experiments/top_k.json", "experiments/hall_of_fame.json"
        )
        data = load_hall_of_fame(Path("experiments/hall_of_fame.json"))
        self._hof = list(data.get("archive", []))
        self.populationChanged.emit()
        self.historyChanged.emit()
        self.paretoChanged.emit()
        self.hallOfFameChanged.emit()

    # ------------------------------------------------------------------
    @Slot()
    def promote(self) -> None:
        """Write the best genome to ``best_config.yaml``."""

        self._ga.promote_best("experiments/best_config.yaml")

    def set_client(self, client: Client, loop: asyncio.AbstractEventLoop) -> None:
        """Attach a WebSocket ``client`` and event ``loop`` for engine integration."""

        self._client = client
        self._loop = loop
        # Use public setter to maintain encapsulation (requires GeneticAlgorithm.set_client)
        self._ga.set_client(client)
        self._ga.set_event_loop(loop)

    def handle_status(self, msg: Dict) -> None:
        """Forward ``ExperimentStatus`` messages to the GA."""

        self._ga.handle_status(msg)

    # ------------------------------------------------------------------
    def _get_population(self) -> List[dict]:
        return self._population

    population = Property("QVariant", _get_population, notify=populationChanged)

    def _get_history(self) -> List[float]:
        return self._history

    history = Property("QVariant", _get_history, notify=historyChanged)

    def _get_pareto(self) -> List[List[float]]:
        return self._pareto

    pareto = Property("QVariant", _get_pareto, notify=paretoChanged)

    def _get_hof(self) -> List[dict]:
        return self._hof

    hallOfFame = Property("QVariant", _get_hof, notify=hallOfFameChanged)
