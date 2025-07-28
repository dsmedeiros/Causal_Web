"""Node-level service classes split from services.py."""

from __future__ import annotations

import uuid
import cmath
import math
from dataclasses import dataclass
from typing import Any
from concurrent.futures import ThreadPoolExecutor
import numpy as np

from ...config import Config
from ..logger import log_json
from ..tick import Tick, GLOBAL_TICK_POOL
from ..node import Node, NodeType, Edge


@dataclass
class NodeTickService:
    """Lifecycle manager for :meth:`~Causal_Web.engine.node.Node.apply_tick`."""

    node: Node
    tick_time: int
    phase: float
    graph: Any
    origin: str = "self"

    def process(self) -> None:
        """Execute the node tick lifecycle for ``tick_time``."""

        if not self._pre_check():
            return
        tick_obj = self._register_tick()
        self._propagate_edges(tick_obj)
        if self.origin == "self":
            collapsed = self.node.propagate_collapse(self.tick_time, self.graph)
            if collapsed:
                self.node._log_collapse_chain(self.tick_time, collapsed)

    # ------------------------------------------------------------------
    def _pre_check(self) -> bool:
        from .. import tick_engine as te

        if self.node.node_type == NodeType.NULL:
            log_json(
                Config.output_path("boundary_interaction_log.json"),
                {"tick": self.tick_time, "void": self.node.id, "origin": self.origin},
            )
            te.void_absorption_events += 1
            self.node._log_tick_drop(self.tick_time, "void_node")
            return False
        if self.node.is_classical:
            print(f"[{self.node.id}] Classical node cannot emit ticks")
            self.node._log_tick_drop(self.tick_time, "classical")
            return False
        if getattr(self.node, "boundary", False):
            log_json(
                Config.output_path("boundary_interaction_log.json"),
                {"tick": self.tick_time, "node": self.node.id, "origin": self.origin},
            )
            te.boundary_interactions_count += 1
        if not te.register_firing(self.node):
            self.node._log_tick_drop(self.tick_time, "bandwidth_limit")
            return False
        if self.origin == "self" and self.tick_time in self.node.emitted_tick_times:
            self.node._log_tick_drop(self.tick_time, "duplicate")
            return False
        return True

    # ------------------------------------------------------------------
    def _register_tick(self) -> Tick:
        trace_id = str(uuid.uuid4())
        tick_obj = GLOBAL_TICK_POOL.acquire()
        tick_obj.origin = self.origin
        tick_obj.time = self.tick_time
        tick_obj.amplitude = 1.0
        tick_obj.phase = self.phase
        tick_obj.layer = "tick"
        tick_obj.trace_id = trace_id

        n = self.node
        with n.lock:
            n.current_tick += 1
            n.subjective_ticks += 1
            n.last_tick_time = self.tick_time
            n.current_threshold = min(n.current_threshold + 0.05, 1.0)
            n.phase = self.phase
            n.tick_history.append(tick_obj)
        log_json(
            Config.output_path("tick_emission_log.json"),
            {"node_id": n.id, "tick_time": self.tick_time, "phase": self.phase},
        )
        with n.lock:
            if self.origin == "self":
                n.emitted_tick_times.add(self.tick_time)
            else:
                n.received_tick_times.add(self.tick_time)
            n._tick_phase_lookup[self.tick_time] = self.phase
        from ..tick_router import TickRouter

        TickRouter.route_tick(n, tick_obj)
        with n.lock:
            n.collapse_origin[self.tick_time] = self.origin
            print(
                f"[{n.id}] Tick at {self.tick_time} via {self.origin.upper()} | Phase: {self.phase:.2f}"
            )
            n._update_memory(self.tick_time, self.origin)
            n._adapt_behavior()
            n.update_node_type()
        return tick_obj

    # ------------------------------------------------------------------
    def _propagate_edges(self, tick: Tick) -> None:
        EdgePropagationService(
            node=self.node,
            tick_time=self.tick_time,
            phase=self.phase,
            origin=self.origin,
            graph=self.graph,
            tick=tick,
        ).propagate()


@dataclass
class EdgePropagationService:
    """Handle tick propagation across outgoing edges.

    Propagation runs in parallel across edges using a thread pool.
    """

    node: Node
    tick_time: int
    phase: float
    origin: str
    graph: Any
    tick: Tick

    def propagate(self) -> None:
        """Propagate the tick across all outgoing edges."""

        self._log_recursion()
        from ..tick_engine import kappa

        edges = self.node._fanout_edges(self.graph)
        with ThreadPoolExecutor(
            max_workers=getattr(Config, "thread_count", None)
        ) as ex:
            ex.map(lambda e: self._propagate_edge(e, kappa), edges)

    # ------------------------------------------------------------------
    def _log_recursion(self) -> None:
        if self.origin != "self" and any(
            e.target == self.origin for e in self.graph.get_edges_from(self.node.id)
        ):
            log_json(
                Config.output_path("refraction_log.json"),
                {
                    "tick": self.tick_time,
                    "recursion_from": self.origin,
                    "node": self.node.id,
                },
            )

    # ------------------------------------------------------------------
    def _propagate_edge(self, edge: Edge, kappa: float) -> None:
        target = self.graph.get_node(edge.target)
        delay = edge.adjusted_delay(
            self.node.law_wave_frequency,
            target.law_wave_frequency,
            kappa,
            graph=self.graph,
        )
        shifted = self._shift_phase(edge)
        self._log_propagation(target, delay, shifted)
        new_delay = self.tick.cumulative_delay + delay
        max_delay = getattr(Config, "max_cumulative_delay", 0)
        if max_delay and new_delay > max_delay:
            if getattr(Config, "log_tick_drops", True):
                target._log_tick_drop(
                    self.tick_time,
                    "event_horizon",
                    tick_id=self.tick.trace_id,
                    source=self.node.id,
                    target=target.id,
                    cumulative_delay=new_delay,
                    coherence=target.compute_coherence_level(self.tick_time),
                )
            return
        if self._handle_refraction(target, delay, shifted, kappa):
            return
        target.schedule_tick(
            self.tick_time + delay,
            shifted,
            origin=self.node.id,
            created_tick=self.tick_time,
            tick_id=self.tick.trace_id,
            cumulative_delay=new_delay,
        )

    # ------------------------------------------------------------------
    def _shift_phase(self, edge: Edge) -> float:
        return self.phase * edge.attenuation + edge.phase_shift

    # ------------------------------------------------------------------
    def _log_propagation(self, target: Node, delay: float, shifted: float) -> None:
        log_json(
            Config.output_path("tick_propagation_log.json"),
            {
                "source": self.node.id,
                "target": target.id,
                "tick_time": self.tick_time,
                "arrival_time": self.tick_time + delay,
                "phase": shifted,
            },
        )

    # ------------------------------------------------------------------
    def _handle_refraction(
        self, target: Node, delay: float, shifted: float, kappa: float
    ) -> bool:
        if target.node_type != NodeType.DECOHERENT:
            return False
        alts = self.graph.get_edges_from(target.id)
        if not alts:
            return False
        alt = alts[0]
        alt_tgt = self.graph.get_node(alt.target)
        alt_delay = alt.adjusted_delay(
            target.law_wave_frequency,
            alt_tgt.law_wave_frequency,
            kappa,
            graph=self.graph,
        )
        alt_tgt.schedule_tick(
            self.tick_time + delay + alt_delay,
            shifted,
            origin=self.node.id,
            created_tick=self.tick_time,
        )
        target.node_type = NodeType.REFRACTIVE
        log_json(
            Config.output_path("refraction_log.json"),
            {
                "tick": self.tick_time,
                "from": self.node.id,
                "via": target.id,
                "to": alt_tgt.id,
            },
        )
        return True


@dataclass
class NodeTickDecisionService:
    """Evaluate whether a node should emit a tick at a given time."""

    node: Node
    tick_time: int

    # ------------------------------------------------------------------
    def decide(self) -> tuple[bool, float | None, str]:
        """Return firing decision, phase and reason."""
        in_refractory = self._is_in_refractory()
        (
            raw_items,
            vector_sum,
            magnitude,
            coherence,
            tick_energy,
        ) = self._phase_metrics()

        if tick_energy < getattr(Config, "tick_threshold", 1):
            return self._below_count(coherence)

        if in_refractory:
            return self._during_refractory(coherence)

        if coherence >= self.node.current_threshold:
            return self._fire_by_threshold(coherence, vector_sum)

        merged, phase = self.node._resolve_interference(
            self.tick_time, raw_items, vector_sum
        )
        if merged:
            return self._fire_by_merge(coherence, phase)

        return self._fail_below_threshold(coherence, magnitude, raw_items)

    # ------------------------------------------------------------------
    def _is_in_refractory(self) -> bool:
        if self.node.current_tick > 0 and self.node.last_tick_time is not None:
            return (
                self.tick_time - self.node.last_tick_time < self.node.refractory_period
            )
        return False

    # ------------------------------------------------------------------
    def _phase_metrics(self) -> tuple[list, complex, float, float, float]:
        raw_items = self.node.incoming_phase_queue[self.tick_time]
        complex_phases = []
        weights = []
        for item in raw_items:
            if isinstance(item, (tuple, list)):
                ph = item[0]
                created = item[1] if len(item) > 1 else Config.current_tick
                decay = getattr(Config, "tick_decay_factor", 1.0) ** (
                    max(0, Config.current_tick - created)
                )
            else:
                ph = item
                decay = 1.0
            complex_phases.append(decay * cmath.rect(1.0, ph % (2 * math.pi)))
            weights.append(decay)
        vector_sum = sum(complex_phases)
        magnitude = abs(vector_sum)
        total_weight = sum(weights) if weights else 0.0
        coherence = magnitude / total_weight if total_weight else 1.0
        tick_energy = total_weight
        return raw_items, vector_sum, magnitude, coherence, tick_energy

    # ------------------------------------------------------------------
    def _log_eval(
        self, coherence: float, refractory: bool, fired: bool, reason: str | None = None
    ) -> None:
        self.node._log_tick_evaluation(
            self.tick_time,
            coherence,
            self.node.current_threshold,
            refractory,
            fired,
            reason,
        )

    # ------------------------------------------------------------------
    def _below_count(self, coherence: float) -> tuple[bool, None, str]:
        self._log_eval(coherence, False, False, "below_count")
        log_json(
            Config.output_path("should_tick_log.json"),
            {"tick": self.tick_time, "node": self.node.id, "reason": "below_count"},
        )
        return False, None, "count_threshold"

    # ------------------------------------------------------------------
    def _during_refractory(self, coherence: float) -> tuple[bool, None, str]:
        self._log_eval(coherence, True, False, "refractory")
        print(f"[{self.node.id}] Suppressed by refractory period at {self.tick_time}")
        return False, None, "refractory"

    # ------------------------------------------------------------------
    def _fire_by_threshold(
        self, coherence: float, vector_sum
    ) -> tuple[bool, float, str]:
        resultant_phase = cmath.phase(vector_sum)
        self._log_eval(coherence, False, True)
        log_json(
            Config.output_path("should_tick_log.json"),
            {"tick": self.tick_time, "node": self.node.id, "reason": "threshold"},
        )
        return True, resultant_phase, "threshold"

    # ------------------------------------------------------------------
    def _fire_by_merge(self, coherence: float, phase: float) -> tuple[bool, float, str]:
        self._log_eval(coherence, False, True, "merged")
        log_json(
            Config.output_path("should_tick_log.json"),
            {"tick": self.tick_time, "node": self.node.id, "reason": "merged"},
        )
        return True, phase, "merged"

    # ------------------------------------------------------------------
    def _fail_below_threshold(
        self, coherence: float, magnitude: float, raw_items
    ) -> tuple[bool, None, str]:
        self._log_eval(coherence, False, False, "below_threshold")
        log_json(
            Config.output_path("magnitude_failure_log.json"),
            {
                "tick": self.tick_time,
                "node": self.node.id,
                "magnitude": round(magnitude, 4),
                "threshold": round(self.node.current_threshold, 4),
                "phases": len(raw_items),
            },
        )
        return False, None, "below_threshold"
