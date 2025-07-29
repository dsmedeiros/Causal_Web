from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .models.node import Node


@dataclass
class MetaNode:
    """Group of nodes subject to optional constraints."""

    member_ids: List[str]
    graph: "CausalGraph"
    meta_type: str = "Configured"
    constraints: Dict[str, Dict] = field(default_factory=dict)
    origin: Optional[str] = None
    collapsed: bool = False
    x: float = 0.0
    y: float = 0.0
    id: str = ""
    phase: float = 0.0

    def __post_init__(self) -> None:
        if not self.id:
            self.id = "meta_" + "_".join(sorted(self.member_ids))

    # ------------------------------------------------------------------
    def apply_tick(self, tick_time: int, phase: float, origin: str = "meta") -> None:
        """Propagate ``phase`` to all member nodes."""
        self.phase = phase
        for nid in self.member_ids:
            node: Optional[Node] = self.graph.get_node(nid)
            if node:
                node.apply_tick(tick_time, phase, self.graph, origin=origin)

    # ------------------------------------------------------------------
    def update_internal_state(self, tick_time: int) -> None:
        """Tick member nodes and enforce constraints."""
        self._enforce_constraints(tick_time)
        for nid in list(self.member_ids):
            node: Optional[Node] = self.graph.get_node(nid)
            if node:
                node.maybe_tick(tick_time, self.graph)

    # ------------------------------------------------------------------
    def _enforce_constraints(self, tick_time: int) -> None:
        """Apply configured MetaNode constraints."""

        if self.meta_type != "Configured":
            return

        cons = self.constraints

        # Phase lock ---------------------------------------------------
        if "phase_lock" in cons:
            tol = float(cons["phase_lock"].get("tolerance", 0.0))
            phases = []
            for nid in self.member_ids:
                node = self.graph.get_node(nid)
                if node:
                    phases.append(node.phase)
            if phases:
                avg = sum(phases) / len(phases)
                for nid in self.member_ids:
                    node = self.graph.get_node(nid)
                    if node and abs(node.phase - avg) > tol:
                        node.phase = avg

        # Shared tick input -------------------------------------------
        if cons.get("shared_tick_input"):
            combined: List = []
            for nid in self.member_ids:
                node = self.graph.get_node(nid)
                if node:
                    combined.extend(node.incoming_phase_queue.get(tick_time, []))
            if combined:
                for nid in self.member_ids:
                    node = self.graph.get_node(nid)
                    if node:
                        q = node.incoming_phase_queue.setdefault(tick_time, [])
                        for item in combined:
                            if item not in q:
                                q.append(item)

        # Coherence tie -----------------------------------------------
        if "coherence_tie" in cons:
            threshold = float(cons["coherence_tie"].get("min_coherence", 0.0))
            remain = []
            for nid in self.member_ids:
                node = self.graph.get_node(nid)
                if node and node.compute_coherence_level(tick_time) >= threshold:
                    remain.append(nid)
            self.member_ids = remain
