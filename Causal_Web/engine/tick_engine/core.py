"""Core simulation loop for the Causal Web."""

from __future__ import annotations

import json
import os
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from ...config import Config
from ..graph import CausalGraph
from ..log_interpreter import run_interpreter
from ..logger import log_json
from ..observer import Observer
from ..tick_seeder import TickSeeder
from . import bridge_manager, evaluator, log_utils


# ---------------------------------------------------------------------------
# Global state shared across modules
# ---------------------------------------------------------------------------

graph = CausalGraph()
observers: list[Observer] = []
kappa = 0.5
seeder = TickSeeder(graph)


def _ensure_attached() -> None:
    evaluator.attach_graph(graph)
    bridge_manager.attach_graph(graph)
    log_utils.attach_graph(graph)


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------


def clear_output_directory() -> None:
    out_dir = Config.output_dir
    if not os.path.isdir(out_dir):
        return
    for name in os.listdir(out_dir):
        if name == "__init__.py":
            continue
        path = os.path.join(out_dir, name)
        if os.path.isfile(path):
            open(path, "w").close()
        elif os.path.isdir(path):
            shutil.rmtree(path)


def build_graph() -> None:
    clear_output_directory()
    graph.load_from_file(Config.input_path("graph.json"))
    global seeder
    seeder = TickSeeder(graph)
    _ensure_attached()


def add_observer(observer: Observer) -> None:
    observers.append(observer)


def emit_ticks(global_tick: int) -> None:
    seeder.seed(global_tick)


def propagate_phases(global_tick: int) -> None:
    pass


# ---------------------------------------------------------------------------
# Simulation state
# ---------------------------------------------------------------------------

_stop_requested = False


def _update_simulation_state(
    paused: bool, stopped: bool, tick: int, snapshot: str | None
) -> None:
    state = {
        "paused": paused,
        "stopped": stopped,
        "current_tick": tick,
        "graph_snapshot": snapshot,
    }
    with open(Config.output_path("simulation_state.json"), "w") as f:
        json.dump(state, f, indent=2)


def pause_simulation() -> None:
    with Config.state_lock:
        Config.is_running = False


def resume_simulation() -> None:
    with Config.state_lock:
        Config.is_running = True


def stop_simulation() -> None:
    global _stop_requested
    with Config.state_lock:
        Config.is_running = False
        _stop_requested = True


# ---------------------------------------------------------------------------
# Simulation runner
# ---------------------------------------------------------------------------


class SimulationRunner:
    """Background worker driving the main simulation loop."""

    def __init__(self) -> None:
        self.global_tick = 0

    def start(self) -> None:
        threading.Thread(target=self.run, daemon=True).start()

    def run(self) -> None:
        global _stop_requested
        _stop_requested = False
        self._seed_random()
        _ensure_attached()
        _update_simulation_state(False, False, self.global_tick, None)
        while True:
            running, stop, rate, limit = self._read_state()
            if stop:
                snapshot = log_utils.snapshot_graph(self.global_tick)
                _update_simulation_state(False, True, self.global_tick, snapshot)
                log_utils.write_output()
                break
            if not running:
                time.sleep(0.1)
                continue
            self._process_tick()
            if limit and limit != -1 and self.global_tick >= limit:
                with Config.state_lock:
                    Config.is_running = False
                snapshot = log_utils.snapshot_graph(self.global_tick)
                _update_simulation_state(False, True, self.global_tick, snapshot)
                log_utils.write_output()
                break
            self.global_tick += 1
            time.sleep(rate)

    def _seed_random(self) -> None:
        if getattr(Config, "random_seed", None) is not None:
            import random

            random.seed(Config.random_seed)
            np.random.seed(Config.random_seed)

    def _read_state(self):
        with Config.state_lock:
            running = Config.is_running
            stop = _stop_requested
            Config.current_tick = self.global_tick
            rate = Config.tick_rate
            limit = (
                Config.max_ticks if Config.allow_tick_override else Config.tick_limit
            )
        return running, stop, rate, limit

    def _process_tick(self) -> None:
        print(f"== Tick {self.global_tick} ==")
        evaluator.apply_global_forcing(self.global_tick)
        evaluator.update_coherence_constraints()
        emit_ticks(self.global_tick)
        propagate_phases(self.global_tick)

        if self.global_tick % getattr(Config, "cluster_interval", 1) == 0:
            graph.detect_clusters()
            evaluator.evaluate_nodes(self.global_tick)
            graph.update_meta_nodes(self.global_tick)
            if not Config.headless:
                log_utils.log_metrics_per_tick(self.global_tick)
                log_utils.log_bridge_states(self.global_tick)
                log_utils.log_meta_node_ticks(self.global_tick)
                log_utils.log_curvature_per_tick(self.global_tick)
            bridge_manager.dynamic_bridge_management(self.global_tick)

        evaluator.check_propagation(self.global_tick)
        snapshot_path = None
        if not Config.headless:
            snapshot_path = log_utils.snapshot_graph(self.global_tick)
        _update_simulation_state(False, False, self.global_tick, snapshot_path)

        if not Config.headless:
            self._handle_observers()

        for bridge in graph.bridges:
            bridge.apply(self.global_tick, graph)

    def _handle_observers(self) -> None:
        for obs in observers:
            obs.observe(graph, self.global_tick)
            inferred = obs.infer_field_state()
            log_json(
                Config.output_path("observer_perceived_field.json"),
                {"tick": self.global_tick, "observer": obs.id, "state": inferred},
            )
            actual = {n.id: len(n.tick_history) for n in graph.nodes.values()}
            diff = {
                nid: {"actual": actual.get(nid, 0), "inferred": inferred.get(nid, 0)}
                for nid in set(actual) | set(inferred)
                if actual.get(nid, 0) != inferred.get(nid, 0)
            }
            if diff:
                log_json(
                    Config.output_path("observer_disagreement_log.json"),
                    {"tick": self.global_tick, "observer": obs.id, "diff": diff},
                )


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def simulation_loop() -> None:
    SimulationRunner().start()
