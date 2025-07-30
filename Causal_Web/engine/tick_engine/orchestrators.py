"""Helper orchestrator classes for the simulation loop."""

from __future__ import annotations

from typing import Callable, List, Optional
from concurrent.futures import ThreadPoolExecutor

from ...config import Config
from ..models.graph import CausalGraph
from ..models.observer import Observer
from ..logging.logger import log_json
from . import bridge_manager, evaluator
from ..logging import log_utils


class EvaluationOrchestrator:
    """Coordinate node evaluation operations."""

    def __init__(self, graph: CausalGraph) -> None:
        self.graph = graph

    def prepare(self, tick: int) -> None:
        evaluator.apply_global_forcing(tick)
        evaluator.update_coherence_constraints()

    def evaluate(self, tick: int) -> None:
        evaluator.evaluate_nodes(tick)

    def finalize(self, tick: int) -> None:
        evaluator.check_propagation(tick)


class MutationOrchestrator:
    """Handle graph mutation logic for each tick."""

    def __init__(
        self,
        graph: CausalGraph,
        emit_func: Callable[[int], None],
        propagate_func: Callable[[int], None],
    ) -> None:
        self.graph = graph
        self._emit = emit_func
        self._propagate = propagate_func

    def pre_process(self, tick: int) -> None:
        self._emit(tick)
        self._propagate(tick)

    def cluster_ops(self, tick: int) -> None:
        self.graph.set_current_tick(tick)
        self.graph.detect_clusters()
        self.graph.update_meta_nodes(tick)
        if (
            tick
            % getattr(Config, "bridge_interval", getattr(Config, "cluster_interval", 1))
            == 0
        ):
            bridge_manager.dynamic_bridge_management(tick)

    def apply_bridges(self, tick: int) -> None:
        """Activate all bridges concurrently for ``tick``."""

        with ThreadPoolExecutor(
            max_workers=getattr(Config, "thread_count", None)
        ) as ex:
            ex.map(lambda b: b.apply(tick, self.graph), self.graph.bridges)


class IOOrchestrator:
    """Manage logging and simulation state updates."""

    def __init__(
        self,
        graph: CausalGraph,
        observers: List[Observer],
        state_updater: Callable[[bool, bool, int, Optional[str]], None],
    ) -> None:
        self.graph = graph
        self.observers = observers
        self._update_state = state_updater

    def log_cluster_info(self, tick: int) -> None:
        if Config.headless:
            return
        interval = getattr(Config, "log_interval", 1)
        if interval and tick % interval != 0:
            return
        log_utils.log_metrics_per_tick(tick)
        log_utils.log_bridge_states(tick)
        log_utils.log_meta_node_ticks(tick)
        log_utils.log_curvature_per_tick(tick)

    def snapshot_state(self, tick: int) -> Optional[str]:
        if Config.headless:
            return None
        return log_utils.snapshot_graph(tick)

    def update_state(
        self, tick: int, paused: bool, stopped: bool, snapshot: Optional[str]
    ) -> None:
        self._update_state(paused, stopped, tick, snapshot)

    def handle_observers(self, tick: int) -> None:
        if Config.headless:
            return
        for obs in self.observers:
            obs.observe(self.graph, tick)
            inferred = obs.infer_field_state()
            log_json(
                "event",
                "observer_perceived_field",
                {"observer": obs.id, "state": inferred},
                tick=tick,
            )
            actual = {n.id: len(n.tick_history) for n in self.graph.nodes.values()}
            diff = {
                nid: {"actual": actual.get(nid, 0), "inferred": inferred.get(nid, 0)}
                for nid in set(actual) | set(inferred)
                if actual.get(nid, 0) != inferred.get(nid, 0)
            }
            if diff:
                log_json(
                    "event",
                    "observer_disagreement_log",
                    {"observer": obs.id, "diff": diff},
                    tick=tick,
                )
