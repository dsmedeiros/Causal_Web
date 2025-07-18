import math
import cmath
from collections import defaultdict, deque
from enum import Enum
from typing import Set, List, Dict, Optional
import numpy as np
import json
import uuid
from ..config import Config
from .logger import log_json
from .tick import Tick, GLOBAL_TICK_POOL


class NodeType(Enum):
    NORMAL = "normal"
    DECOHERENT = "decoherent"
    CLASSICALIZED = "classicalized"
    ENTANGLED = "entangled"
    REFRACTIVE = "refractive"
    NULL = "null"


class Node:
    """Represents a single oscillator in the causal graph."""

    def __init__(
        self,
        node_id,
        x=0.0,
        y=0.0,
        frequency=1.0,
        refractory_period=2,
        base_threshold=0.5,
        phase=0.0,
        *,
        origin_type: str = "seed",
        generation_tick: int = 0,
        parent_ids: Optional[List[str]] = None,
    ):
        self.id = node_id
        self.x = x
        self.y = y
        self.frequency = frequency
        self.phase = phase
        self.coherence = 1.0
        self.decoherence = 0.0
        self.tick_history = []  # List[Tick]
        # Separate tracking for self-emitted vs externally sourced ticks
        self.emitted_tick_times: Set[float] = set()
        self.received_tick_times: Set[float] = set()
        self._tick_phase_lookup: Dict[int, float] = {}
        self.incoming_phase_queue = defaultdict(list)  # tick_time -> [phase_i]
        self.pending_superpositions = defaultdict(list)  # for logging and analysis
        self._phase_cache: Dict[int, float] = {}
        self._coherence_cache: Dict[int, float] = {}
        self._decoherence_cache: Dict[int, float] = {}
        self.current_tick = 0
        self.subjective_ticks = 0  # For relativistic tracking
        self.last_emission_tick = None
        self.refractory_period = refractory_period
        self.last_tick_time: Optional[float] = None
        self.base_threshold = base_threshold
        self.current_threshold = self.base_threshold
        self.collapse_origin = {}  # tick_time -> "self" or "bridge"
        self._decoherence_streak = 0
        self.is_classical = False
        self.coherence_series: List[float] = []
        self.law_wave_frequency: float = 0.0
        self.entangled_with: Set[str] = set()
        self.coherence_velocity: float = 0.0
        self.node_type: NodeType = NodeType.NORMAL
        self.prev_node_type: NodeType = NodeType.NORMAL
        self.coherence_credit: float = 0.0
        self.decoherence_debt: float = 0.0
        self.phase_lock: bool = False
        self.collapse_pressure: float = 0.0

        # ---- Cluster metadata ----
        # Mapping of hierarchy level -> cluster id
        self.cluster_ids: Dict[int, int] = {}

        # ---- Propagation metadata ----
        self.origin_type = origin_type
        self.generation_tick = generation_tick
        self.parent_ids = parent_ids or []
        self.sip_streak = 0

        # ---- Phase 4 additions ----
        self.memory_window = getattr(Config, "memory_window", 20)
        self.memory: Dict[str, deque] = {
            "origins": deque(maxlen=self.memory_window),
            "coherence": deque(maxlen=self.memory_window),
            "decoherence": deque(maxlen=self.memory_window),
        }
        self.trust_profile: Dict[str, float] = {}
        self.phase_confidence_index: float = 1.0
        self.goals: Dict[str, float] = {}
        self.goal_error: Dict[str, float] = {}

        # ---- Coherence threshold ramp parameters ----
        self.initial_coherence_threshold = getattr(
            Config, "initial_coherence_threshold", 0.6
        )
        self.steady_coherence_threshold = getattr(
            Config, "steady_coherence_threshold", 0.85
        )
        self.coherence_ramp_ticks = getattr(Config, "coherence_ramp_ticks", 10)

        # additional offset applied from global network constraints
        self.dynamic_offset = 0.0

        # ---- Spatial indexing ----
        cell_size = getattr(Config, "SPATIAL_GRID_SIZE", 50)
        self.grid_x = int(self.x // cell_size)
        self.grid_y = int(self.y // cell_size)

    def compute_phase(self, tick_time):
        """Return phase value incorporating time-dependent global jitter."""
        if tick_time in self._phase_cache:
            return self._phase_cache[tick_time]
        base = 2 * math.pi * self.frequency * tick_time
        jitter = Config.phase_jitter
        if jitter["amplitude"] and jitter.get("period", 0):
            ramp = min(tick_time / max(1, Config.forcing_ramp_ticks), 1.0)
            base += (
                ramp
                * jitter["amplitude"]
                * math.sin(2 * math.pi * tick_time / jitter["period"])
            )
        self._phase_cache[tick_time] = base
        return base

    def _coherence_threshold(self) -> float:
        """Return dynamic coherence acceptance threshold."""
        progress = (
            min(self.subjective_ticks, self.coherence_ramp_ticks)
            / self.coherence_ramp_ticks
        )
        base = (
            self.initial_coherence_threshold
            + (self.steady_coherence_threshold - self.initial_coherence_threshold)
            * progress
        )
        return base + self.dynamic_offset

    def compute_coherence_level(self, tick_time):
        if tick_time in self._coherence_cache:
            return self._coherence_cache[tick_time]
        phases = self.pending_superpositions.get(tick_time, [])
        old = self.coherence
        if len(phases) < 2:
            self.coherence = 1.0
            self._update_law_wave()
            self.coherence_credit += 0.1
            self.decoherence_debt = max(0.0, self.decoherence_debt - 0.1)
            new = self.coherence
            if abs(new - old) > 0.05:
                from . import tick_engine as te

                te.mark_for_update(self.id)
            return self.coherence
        complex_vecs = [cmath.rect(1.0, p % (2 * math.pi)) for p in phases]
        vector_sum = sum(complex_vecs)
        self.coherence = abs(vector_sum) / len(phases)
        threshold = self._coherence_threshold()
        if self.coherence > threshold:
            self.coherence_credit += self.coherence - threshold
        else:
            self.decoherence_debt += 0.1
        self._update_law_wave()
        if abs(self.coherence - old) > 0.05:
            from . import tick_engine as te

            te.mark_for_update(self.id)
        self._coherence_cache[tick_time] = self.coherence
        return self.coherence

    def compute_decoherence_field(self, tick_time):
        if tick_time in self._decoherence_cache:
            return self._decoherence_cache[tick_time]
        phases = self.pending_superpositions.get(tick_time, [])
        old = self.decoherence
        if len(phases) < 2:
            self.decoherence = 0.0
            self.decoherence_debt = max(0.0, self.decoherence_debt - 0.1)
            if abs(self.decoherence - old) > 0.05:
                from . import tick_engine as te

                te.mark_for_update(self.id)
            return self.decoherence
        normalized = [(p % (2 * math.pi)) for p in phases]
        mean_phase = sum(normalized) / len(normalized)
        variance = sum((p - mean_phase) ** 2 for p in normalized) / len(normalized)
        self.decoherence = variance
        if self.decoherence > 0.4:
            self.decoherence_debt += self.decoherence - 0.4
        else:
            self.coherence_credit += 0.05
        if abs(self.decoherence - old) > 0.05:
            from . import tick_engine as te

            te.mark_for_update(self.id)
        self._decoherence_cache[tick_time] = self.decoherence
        return self.decoherence

    # ------------------------------------------------------------------
    def _update_memory(self, tick_time: int, origin: Optional[str] = None) -> None:
        coherence = self.compute_coherence_level(tick_time)
        decoherence = self.compute_decoherence_field(tick_time)
        self.memory["coherence"].append(coherence)
        self.memory["decoherence"].append(decoherence)
        if origin is not None:
            self.memory["origins"].append(origin)
            score = self.trust_profile.get(origin, 0.5)
            threshold = self._coherence_threshold()
            if coherence > threshold:
                score = min(1.0, score + 0.05)
            else:
                score = max(0.0, score - 0.05)
            self.trust_profile[origin] = score
        if len(self.memory["coherence"]) > 1:
            self.phase_confidence_index = 1.0 - float(np.var(self.memory["coherence"]))
        else:
            self.phase_confidence_index = 1.0

        if coherence > self._coherence_threshold() + 0.1:
            self.sip_streak += 1
        else:
            self.sip_streak = 0

    def _adapt_behavior(self) -> None:
        old_thresh = self.current_threshold
        if self.memory["coherence"]:
            avg_coh = sum(self.memory["coherence"]) / len(self.memory["coherence"])
        else:
            avg_coh = 1.0

        # adjust thresholds based on recent stability
        if avg_coh > 0.9:
            self.current_threshold = max(
                self.base_threshold * 0.8, self.current_threshold - 0.01
            )
            self.refractory_period = max(1.0, self.refractory_period - 0.05)
        elif avg_coh < 0.5:
            self.current_threshold = min(1.0, self.current_threshold + 0.02)
            self.refractory_period += 0.05

        goal = self.goals.get("coherence")
        if goal is not None:
            error = goal - avg_coh
            self.goal_error["coherence"] = error
            self.current_threshold -= error * 0.05
        if abs(self.current_threshold - old_thresh) > 0.01:
            from . import tick_engine as te

            te.mark_for_update(self.id)

    def _phase_drift_tolerance(self, tick_time: int) -> float:
        """Return allowable phase drift during early formation."""
        ramp = getattr(Config, "DRIFT_TOLERANCE_RAMP", 10)
        progress = min(tick_time, ramp) / ramp
        return (math.pi / 2) * (1 - progress) + 0.1 * progress

    def _fanout_edges(self, graph) -> list:
        """Return outbound edges limited by ``Config.max_tick_fanout``."""
        edges = graph.get_edges_from(self.id)
        # limit propagation to nodes within the same primary cluster
        cluster = self.cluster_ids.get(0)
        if cluster is not None:
            # only enforce cluster restriction when the cluster contains
            # multiple nodes. During early formation each node may be in its
            # own cluster, which would otherwise block all outbound edges.
            cluster_size = sum(
                1 for n in graph.nodes.values() if n.cluster_ids.get(0) == cluster
            )
            if cluster_size > 1:
                edges = [
                    e
                    for e in edges
                    if graph.get_node(e.target).cluster_ids.get(0) == cluster
                ]
        limit = getattr(Config, "max_tick_fanout", 0)
        if limit and len(edges) > limit:
            return edges[:limit]
        return edges

    def _resolve_interference(
        self, tick_time: int, raw_phases: list, vector_sum: complex
    ):
        """Attempt to merge near-aligned phases into a single tick."""
        tol = self._phase_drift_tolerance(tick_time)
        diffs = [
            abs((p1 - p2 + math.pi) % (2 * math.pi) - math.pi)
            for i, p1 in enumerate(raw_phases)
            for p2 in raw_phases[i + 1 :]
        ]
        if diffs and max(diffs) < tol:
            return True, cmath.phase(vector_sum)
        if abs(vector_sum) >= self.current_threshold * 0.8:
            return True, cmath.phase(vector_sum)
        return False, None

    def _update_law_wave(self, window: int = 20):
        """Compute dominant coherence frequency over a sliding window."""
        self.coherence_series.append(self.coherence)
        if len(self.coherence_series) > window:
            self.coherence_series.pop(0)
        if len(self.coherence_series) > 1:
            arr = np.array(self.coherence_series[-window:])
            spectrum = np.fft.rfft(arr)
            if spectrum.size > 1:
                idx = int(np.argmax(np.abs(spectrum[1:])) + 1)
                self.law_wave_frequency = idx / window

    def update_classical_state(
        self,
        decoherence_strength,
        tick_time=None,
        graph=None,
        threshold=0.4,
        streak_required=2,
    ):
        if decoherence_strength > threshold:
            self._decoherence_streak += 1
        else:
            self._decoherence_streak = 0

        if self._decoherence_streak >= streak_required and not self.is_classical:
            self.is_classical = True
            if tick_time is not None:
                record = {"tick": tick_time, "node": self.id, "event": "collapse_start"}
                log_json(Config.output_path("collapse_front_log.json"), record)
                if graph is not None:
                    graph.emit_law_wave(self.id, tick_time)
            self.update_node_type()

    def update_node_type(self) -> None:
        old = self.node_type
        if self.is_classical:
            self.node_type = NodeType.CLASSICALIZED
            self.phase_lock = True
        elif self.decoherence_debt > 3.0:
            self.node_type = NodeType.DECOHERENT
        elif self.entangled_with:
            self.node_type = NodeType.ENTANGLED
        else:
            self.node_type = NodeType.NORMAL

        if old != self.node_type:
            rec = {"node": self.id, "from": old.value, "to": self.node_type.value}
            log_json(Config.output_path("node_state_map.json"), rec)
        self.prev_node_type = old

    def _log_tick_evaluation(
        self,
        tick_time: int,
        coherence: float,
        threshold: float,
        refractory: bool,
        fired: bool,
        reason: str | None = None,
    ) -> None:
        record = {
            "tick": tick_time,
            "node": self.id,
            "coherence": round(coherence, 4),
            "threshold": round(threshold, 4),
            "refractory": refractory,
            "fired": fired,
        }
        if reason is not None:
            record["reason"] = reason
        log_json(Config.output_path("tick_evaluation_log.json"), record)
        if not fired:
            fail_rec = {
                "tick": tick_time,
                "node": self.id,
                "threshold": round(threshold, 4),
                "coherence": round(coherence, 4),
                "reason": reason or ("refractory" if refractory else "below_threshold"),
            }
            log_json(Config.output_path("propagation_failure_log.json"), fail_rec)

    def _log_tick_drop(self, tick_time: int, reason: str) -> None:
        record = {
            "tick": tick_time,
            "node": self.id,
            "reason": reason,
            "coherence": round(getattr(self, "coherence", 0.0), 4),
            "node_type": self.node_type.value,
        }
        log_json(Config.output_path("tick_drop_log.json"), record)
        fail_rec = {
            "tick": tick_time,
            "node": self.id,
            "threshold": round(self.current_threshold, 4),
            "coherence": round(getattr(self, "coherence", 0.0), 4),
            "reason": reason,
        }
        log_json(Config.output_path("propagation_failure_log.json"), fail_rec)

    def schedule_tick(self, tick_time, incoming_phase, origin=None):
        """Store an incoming phase for future evaluation.

        Parameters
        ----------
        tick_time : float
            Global tick time at which the phase should be evaluated.
        incoming_phase : float
            Phase value being delivered.
        origin : str, optional
            ID of the node that emitted the phase.
        """

        self.incoming_phase_queue[tick_time].append(incoming_phase)
        self.pending_superpositions[tick_time].append(incoming_phase)
        self._coherence_cache.pop(tick_time, None)
        self._decoherence_cache.pop(tick_time, None)
        print(
            f"[{self.id}] Received tick at {tick_time} with phase {incoming_phase:.2f}"
        )
        if origin is not None:
            log_json(
                Config.output_path("tick_delivery_log.json"),
                {
                    "source": origin,
                    "node_id": self.id,
                    "tick_time": tick_time,
                    "stored_phase": incoming_phase,
                },
            )
        from . import tick_engine as te

        te.mark_for_update(self.id)

    def should_tick(self, tick_time):
        """Determine if the node should emit a tick at ``tick_time``.

        Returns a tuple ``(fire, phase, reason)`` where ``fire`` indicates
        whether a tick should be emitted, ``phase`` is the resultant phase if a
        tick is fired, and ``reason`` provides context for logging when ``fire``
        is ``False``.
        """

        in_refractory = False
        if self.current_tick > 0 and self.last_tick_time is not None:
            in_refractory = tick_time - self.last_tick_time < self.refractory_period
        if self.current_tick == 0:
            in_refractory = False
        raw_phases = self.incoming_phase_queue[tick_time]
        complex_phases = [cmath.rect(1.0, p % (2 * math.pi)) for p in raw_phases]
        vector_sum = sum(complex_phases)
        magnitude = abs(vector_sum)
        coherence = magnitude / len(raw_phases) if raw_phases else 1.0

        if in_refractory:
            self._log_tick_evaluation(
                tick_time,
                coherence,
                self.current_threshold,
                True,
                False,
                "refractory",
            )
            print(f"[{self.id}] Suppressed by refractory period at {tick_time}")
            return False, None, "refractory"

        if coherence >= self.current_threshold:
            resultant_phase = cmath.phase(vector_sum)
            self._log_tick_evaluation(
                tick_time,
                coherence,
                self.current_threshold,
                False,
                True,
            )
            log_json(
                Config.output_path("should_tick_log.json"),
                {"tick": tick_time, "node": self.id, "reason": "threshold"},
            )
            return True, resultant_phase, "threshold"

        merged, phase = self._resolve_interference(tick_time, raw_phases, vector_sum)
        if merged:
            self._log_tick_evaluation(
                tick_time,
                coherence,
                self.current_threshold,
                False,
                True,
                "merged",
            )
            log_json(
                Config.output_path("should_tick_log.json"),
                {"tick": tick_time, "node": self.id, "reason": "merged"},
            )
            return True, phase, "merged"

        self._log_tick_evaluation(
            tick_time,
            coherence,
            self.current_threshold,
            False,
            False,
            "below_threshold",
        )
        log_json(
            Config.output_path("magnitude_failure_log.json"),
            {
                "tick": tick_time,
                "node": self.id,
                "magnitude": round(magnitude, 4),
                "threshold": round(self.current_threshold, 4),
                "phases": len(raw_phases),
            },
        )
        return False, None, "below_threshold"

    def get_phase_at(self, tick_time):
        return self._tick_phase_lookup.get(tick_time)

    def apply_tick(self, tick_time, phase, graph, origin="self"):
        """Emit a tick and propagate resulting phases to neighbours."""

        # Boundary and state checks -------------------------------------------------
        if self.node_type == NodeType.NULL:
            log_json(
                Config.output_path("boundary_interaction_log.json"),
                {"tick": tick_time, "void": self.id, "origin": origin},
            )
            from . import tick_engine as te

            te.void_absorption_events += 1
            self._log_tick_drop(tick_time, "void_node")
            return

        if self.is_classical:
            # Collapsed nodes do not emit further ticks
            print(f"[{self.id}] Classical node cannot emit ticks")
            self._log_tick_drop(tick_time, "classical")
            return

        # Log any interactions with boundary nodes
        if getattr(self, "boundary", False):
            log_json(
                Config.output_path("boundary_interaction_log.json"),
                {"tick": tick_time, "node": self.id, "origin": origin},
            )
            from . import tick_engine as te

            te.boundary_interactions_count += 1

        # Avoid duplicate self-emissions, but allow external ticks separately
        if origin == "self" and tick_time in self.emitted_tick_times:
            self._log_tick_drop(tick_time, "duplicate")
            return

        # Register the emission
        self.current_tick += 1
        self.subjective_ticks += 1
        self.last_tick_time = tick_time
        # Each tick slightly increases threshold making firing harder
        self.current_threshold = min(self.current_threshold + 0.05, 1.0)
        self.phase = phase
        trace_id = str(uuid.uuid4())
        tick_obj = GLOBAL_TICK_POOL.acquire()
        tick_obj.origin = origin
        tick_obj.time = tick_time
        tick_obj.amplitude = 1.0
        tick_obj.phase = phase
        tick_obj.layer = "tick"
        tick_obj.trace_id = trace_id
        self.tick_history.append(tick_obj)
        log_json(
            Config.output_path("tick_emission_log.json"),
            {"node_id": self.id, "tick_time": tick_time, "phase": phase},
        )
        if origin == "self":
            self.emitted_tick_times.add(tick_time)
        else:
            self.received_tick_times.add(tick_time)
        self._tick_phase_lookup[tick_time] = phase
        from .tick_router import TickRouter

        TickRouter.route_tick(self, tick_obj)
        self.collapse_origin[tick_time] = origin
        print(
            f"[{self.id}] Tick at {tick_time} via {origin.upper()} | Phase: {phase:.2f}"
        )

        # Update memory and adapt behaviour
        self._update_memory(tick_time, origin)
        self._adapt_behavior()
        self.update_node_type()

        if origin != "self" and any(
            e.target == origin for e in graph.get_edges_from(self.id)
        ):
            log_json(
                Config.output_path("refraction_log.json"),
                {"tick": tick_time, "recursion_from": origin, "node": self.id},
            )

        # Recursive phase propagation with refraction
        for edge in self._fanout_edges(graph):
            target = graph.get_node(edge.target)
            from .tick_engine import kappa

            delay = edge.adjusted_delay(
                self.law_wave_frequency,
                target.law_wave_frequency,
                kappa,
            )
            attenuated = phase * edge.attenuation
            shifted = attenuated + edge.phase_shift

            log_json(
                Config.output_path("tick_propagation_log.json"),
                {
                    "source": self.id,
                    "target": target.id,
                    "tick_time": tick_time,
                    "arrival_time": tick_time + delay,
                    "phase": shifted,
                },
            )

            if target.node_type == NodeType.DECOHERENT:
                # Redirect through alternative path when hitting a decoherent node
                alts = graph.get_edges_from(target.id)
                if alts:
                    alt = alts[0]
                    alt_tgt = graph.get_node(alt.target)
                    alt_delay = alt.adjusted_delay(
                        target.law_wave_frequency, alt_tgt.law_wave_frequency, kappa
                    )
                    alt_tgt.schedule_tick(
                        tick_time + delay + alt_delay, shifted, origin=self.id
                    )
                    target.node_type = NodeType.REFRACTIVE
                    log_json(
                        Config.output_path("refraction_log.json"),
                        {
                            "tick": tick_time,
                            "from": self.id,
                            "via": target.id,
                            "to": alt_tgt.id,
                        },
                    )
                    continue

            target.schedule_tick(tick_time + delay, shifted, origin=self.id)

        if origin == "self":
            collapsed = self.propagate_collapse(tick_time, graph)
            if collapsed:
                self._log_collapse_chain(tick_time, collapsed)

    def propagate_collapse(
        self, tick_time, graph, threshold: float = 0.5, depth: int = 1, visited=None
    ):
        """Recursively propagate collapse and track chain depth."""
        if visited is None:
            visited = set()
        visited.add(self.id)

        chain = []
        for nid in self.entangled_with:
            if nid in visited:
                continue
            other = graph.get_node(nid)
            if not other or other.collapse_origin.get(tick_time):
                continue
            deco = other.compute_decoherence_field(tick_time)
            if deco > threshold:
                other.apply_tick(tick_time, self.phase, graph, origin="entanglement")
                chain.append({"node": nid, "depth": depth})
                chain.extend(
                    other.propagate_collapse(
                        tick_time, graph, threshold, depth + 1, visited
                    )
                )
        if chain:
            record = {"tick": tick_time, "source": self.id, "chain": chain}
            log_json(Config.output_path("collapse_front_log.json"), record)
        return chain

    def _log_collapse_chain(self, tick_time, collapsed):
        """Log collapse propagation events to file with depth info."""
        record = {
            "tick": tick_time,
            "source": self.id,
            "collapsed": collapsed,
            "collapsed_entity": self.id,
            "children_spawned": [c.get("node") for c in collapsed],
        }
        log_json(Config.output_path("collapse_chain_log.json"), record)

    def maybe_tick(self, global_tick, graph):
        """Evaluate queued phases and emit a tick if conditions are met."""
        if self.tick_history:
            from .tick_router import TickRouter

            TickRouter.route_tick(self, self.tick_history[-1])

        tick_key = global_tick
        if tick_key not in self.incoming_phase_queue:
            for k in list(self.incoming_phase_queue.keys()):
                if math.isclose(k, global_tick, abs_tol=1e-6):
                    tick_key = k
                    break
        if tick_key in self.incoming_phase_queue:
            should_fire, phase, reason = self.should_tick(tick_key)
            if should_fire and not self.is_classical:
                self.apply_tick(tick_key, phase, graph)
            else:
                drop_reason = "classical" if self.is_classical else reason
                self._log_tick_drop(tick_key, drop_reason)
                print(f"[{self.id}] {drop_reason} at {tick_key} cancelled tick")
            del self.incoming_phase_queue[tick_key]
            if tick_key in self.pending_superpositions:
                del self.pending_superpositions[tick_key]
            self._coherence_cache.pop(tick_key, None)
            self._decoherence_cache.pop(tick_key, None)
            self._update_memory(tick_key)
            self._adapt_behavior()
            self.update_node_type()
        else:
            self.current_threshold = max(
                self.base_threshold, self.current_threshold - 0.01
            )

    def _emit(self, tick_time):
        phase = self.compute_phase(tick_time)
        tick_obj = GLOBAL_TICK_POOL.acquire()
        tick_obj.origin = "self"
        tick_obj.time = tick_time
        tick_obj.amplitude = 1.0
        tick_obj.phase = phase
        tick_obj.layer = "tick"
        tick_obj.trace_id = str(uuid.uuid4())
        self.tick_history.append(tick_obj)
        self._tick_phase_lookup[tick_time] = phase
        print(f"[{self.id}] Emitted tick at {tick_time} | Phase: {phase:.2f}")


class Edge:
    """Directional connection carrying phase between nodes."""

    def __init__(self, source, target, attenuation, density, delay=1, phase_shift=0.0):
        self.source = source
        self.target = target
        self.attenuation = attenuation  # Multiplier for phase amplitude
        self.density = density  # Can affect delay dynamically
        self.delay = delay
        self.phase_shift = phase_shift

    def adjusted_delay(
        self, source_freq: float = 0.0, target_freq: float = 0.0, kappa: float = 1.0
    ):
        """Delay adjusted by local law-wave gradient."""
        base = self.delay + int(self.density)
        delta_f = abs(source_freq - target_freq)
        modifier = kappa * math.sin(2 * math.pi * delta_f) * self.density
        adjusted = base + modifier
        # ensure delay remains positive to avoid scheduling errors
        return max(1, int(round(adjusted)))

    def propagate_phase(self, phase, global_tick, graph):
        target_node = graph.get_node(self.target)

        drift_phase = 0.0
        if isinstance(self.phase_shift, dict):
            mode = self.phase_shift.get("mode")
            if mode == "linear":
                rate = self.phase_shift.get("rate", 0.1)
                drift_phase = rate * global_tick
            elif mode == "nonlinear":
                amp = self.phase_shift.get("amplitude", math.pi)
                period = self.phase_shift.get("period", 60)
                drift_phase = amp * math.sin((2 * math.pi / period) * global_tick)
            else:
                raise ValueError(f"Unknown phase shift mode: {mode}")
        else:
            drift_phase = self.phase_shift  # static phase shift

        shifted_phase = phase + drift_phase
        attenuated_phase = shifted_phase * self.attenuation
        scheduled_tick = global_tick + self.adjusted_delay(
            (
                graph.get_node(self.source).law_wave_frequency
                if hasattr(self, "source")
                else 0.0
            ),
            target_node.law_wave_frequency,
        )
        target_node.schedule_tick(scheduled_tick, attenuated_phase, origin=self.source)
