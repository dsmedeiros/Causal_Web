import json
import random
from typing import Dict, List

from .graph import CausalGraph
from ..config import Config


class TickSeeder:
    """Injects ticks into the graph under configurable strategies."""

    def __init__(self, graph: CausalGraph, config: Dict = None, log_path: str = None):
        self.graph = graph
        self.config = config or getattr(Config, "seeding", {"strategy": "static"})
        self.log_path = log_path or Config.output_path("tick_seed_log.json")
        self.seed_count = 0

    # ------------------------------------------------------------
    def seed(self, global_tick: int) -> None:
        strat = self.config.get("strategy", "static")
        if strat == "static":
            self._seed_static(global_tick)
        elif strat == "probabilistic":
            self._seed_probabilistic(global_tick)

    # ------------------------------------------------------------
    def _log(self, record: Dict) -> None:
        with open(self.log_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        self.seed_count += 1

    # ------------------------------------------------------------
    def _seed_static(self, global_tick: int) -> None:
        for source in getattr(self.graph, "tick_sources", []):
            if isinstance(source, str):
                node_id = source
                interval = 1
                phase = 0.0
            elif isinstance(source, dict):
                node_id = source.get("node_id")
                interval = source.get("tick_interval", 1)
                phase = source.get("phase", 0.0)
            else:
                continue

            node = self.graph.get_node(node_id)
            if node and not node.is_classical and global_tick % interval == 0:
                node.apply_tick(global_tick, phase, self.graph, origin="seed")
                self._log({
                    "tick": global_tick,
                    "node": node.id,
                    "phase": phase,
                    "strategy": "static",
                })

    # ------------------------------------------------------------
    def _seed_probabilistic(self, global_tick: int) -> None:
        prob = self.config.get("probability", 0.1)
        nodes = list(self.graph.nodes.values())
        for node in nodes:
            if node.is_classical:
                continue
            if random.random() < prob:
                phase = node.compute_phase(global_tick)
                node.apply_tick(global_tick, phase, self.graph, origin="seed")
                self._log({
                    "tick": global_tick,
                    "node": node.id,
                    "phase": phase,
                    "strategy": "probabilistic",
                })
