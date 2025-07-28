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
from .orchestrators import EvaluationOrchestrator, MutationOrchestrator, IOOrchestrator


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
    graph.load_from_file(Config.graph_file)
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
        self.evaluation = EvaluationOrchestrator(graph)
        self.mutation = MutationOrchestrator(graph, emit_ticks, propagate_phases)
        self.io = IOOrchestrator(graph, observers, _update_simulation_state)

    def start(self) -> None:
        threading.Thread(target=self.run, daemon=True).start()

    def run(self) -> None:
        """Main processing loop executed in a background thread."""
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
            if limit and limit != -1 and self.global_tick >= limit:
                with Config.state_lock:
                    Config.is_running = False
                snapshot = log_utils.snapshot_graph(self.global_tick)
                _update_simulation_state(False, True, self.global_tick, snapshot)
                log_utils.write_output()
                break
            if not running:
                time.sleep(0.1)
                continue
            self._process_tick()
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
        self.evaluation.prepare(self.global_tick)
        self.mutation.pre_process(self.global_tick)

        cluster_cycle = self.global_tick % getattr(Config, "cluster_interval", 1) == 0
        if cluster_cycle:
            self.mutation.cluster_ops(self.global_tick)
            self.evaluation.evaluate(self.global_tick)
            self.io.log_cluster_info(self.global_tick)

        self.evaluation.finalize(self.global_tick)
        snapshot_path = self.io.snapshot_state(self.global_tick)
        self.io.update_state(self.global_tick, False, False, snapshot_path)
        self.io.handle_observers(self.global_tick)
        self.mutation.apply_bridges(self.global_tick)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def simulation_loop() -> None:
    SimulationRunner().start()
