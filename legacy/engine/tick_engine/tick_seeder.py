import json
import random
from typing import Dict, List, Optional

from ..models.graph import CausalGraph
from ...config import Config
from ..logging.logger import log_record


class TickSeeder:
    """Injects ticks into the graph under configurable strategies."""

    def __init__(
        self,
        graph: CausalGraph,
        config: Optional[Dict] = None,
        log_path: Optional[str] = None,
    ) -> None:
        """Create a new seeder for *graph* with optional configuration."""
        self.graph = graph
        self.config = config or getattr(Config, "seeding", {"strategy": "static"})
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
        log_record(
            category="event",
            label="tick_seed_log",
            tick=record.get("tick"),
            value={k: v for k, v in record.items() if k != "tick"},
        )
        self.seed_count += 1

    # ------------------------------------------------------------
    def _seed_static(self, global_tick: int) -> None:
        phase_offsets = self.config.get("phase_offsets", {})
        for source in getattr(self.graph, "tick_sources", []):
            if isinstance(source, str):
                node_id = source
                interval = 1
                phase = 0.0
                offset = phase_offsets.get(node_id)
            elif isinstance(source, dict):
                node_id = source.get("node_id")
                interval = source.get("tick_interval", 1)
                phase = source.get("phase", 0.0)
                offset = source.get("phase_offset")
                if offset is None:
                    offset = phase_offsets.get(node_id)
            else:
                continue

            node = self.graph.get_node(node_id)
            if node and not node.is_classical and global_tick % interval == 0:
                final_phase = (
                    node.compute_phase(global_tick) + offset
                    if offset is not None
                    else phase
                )
                node.apply_tick(global_tick, final_phase, self.graph, origin="seed")
                coherence = getattr(node, "coherence", 0.0)
                threshold = node._coherence_threshold()
                success = coherence >= threshold
                reason = (
                    None
                    if success
                    else f"coherence {coherence:.3f} below {threshold:.3f}"
                )
                self._log(
                    {
                        "tick": global_tick,
                        "node": node.id,
                        "phase": final_phase,
                        "strategy": "static",
                        "coherence": round(coherence, 4),
                        "threshold": round(threshold, 4),
                        "success": success,
                        "failure_reason": reason,
                    }
                )

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
                coherence = getattr(node, "coherence", 0.0)
                threshold = node._coherence_threshold()
                success = coherence >= threshold
                reason = (
                    None
                    if success
                    else f"coherence {coherence:.3f} below {threshold:.3f}"
                )
                self._log(
                    {
                        "tick": global_tick,
                        "node": node.id,
                        "phase": phase,
                        "strategy": "probabilistic",
                        "coherence": round(coherence, 4),
                        "threshold": round(threshold, 4),
                        "success": success,
                        "failure_reason": reason,
                    }
                )
