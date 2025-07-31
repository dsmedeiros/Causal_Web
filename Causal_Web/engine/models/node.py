from __future__ import annotations

import math
import cmath
from collections import defaultdict, deque
from enum import Enum
from typing import Set, List, Dict, Optional
import numpy as np
import json
import uuid
from ...config import Config
from .base import LoggingMixin
from .tick import Tick, GLOBAL_TICK_POOL


class NodeType(Enum):
    NORMAL = "normal"
    DECOHERENT = "decoherent"
    CLASSICALIZED = "classicalized"
    ENTANGLED = "entangled"
    REFRACTIVE = "refractive"
    NULL = "null"


class Node(LoggingMixin):
    """Represents a single oscillator in the causal graph."""

    def __init__(
        self,
        node_id: str,
        x: float = 0.0,
        y: float = 0.0,
        frequency: float = 1.0,
        refractory_period: float | None = None,
        base_threshold: float = 0.5,
        phase: float = 0.0,
        *,
        origin_type: str = "seed",
        generation_tick: int = 0,
        parent_ids: Optional[List[str]] = None,
    ) -> None:
        """Create a new node and delegate attribute setup."""

        from ..services.node_services import NodeInitializationService

        NodeInitializationService(self).setup(
            node_id,
            x=x,
            y=y,
            frequency=frequency,
            refractory_period=refractory_period,
            base_threshold=base_threshold,
            phase=phase,
            origin_type=origin_type,
            generation_tick=generation_tick,
            parent_ids=parent_ids,
        )

    def _advance_internal_phase(self, tick_time: float) -> None:
        """Update :attr:`internal_phase` using entrainment from ticks."""

        dt = tick_time - getattr(self, "_last_phase_update", 0.0)
        base = self.internal_phase + self.frequency * dt
        items = self.incoming_phase_queue.get(tick_time, [])
        vector = 0j
        for item in items:
            ph = item[0]
            amp = item[1] if len(item) > 1 else 1.0
            created = item[2] if len(item) > 2 else Config.current_tick
            decay = getattr(Config, "tick_decay_factor", 1.0) ** (
                max(0, Config.current_tick - created)
            )
            vector += amp * decay * cmath.exp(1j * ph)
        contribution = cmath.phase(vector) if items else 0.0
        new_phase = base + contribution
        if getattr(Config, "smooth_phase", False):
            alpha = getattr(Config, "phase_smoothing_alpha", 0.1)
            self.internal_phase = self.internal_phase * (1 - alpha) + new_phase * alpha
        else:
            self.internal_phase = new_phase
        self._last_phase_update = tick_time

    def compute_phase(self, tick_time: float) -> float:
        """Return the oscillator phase after applying entrainment."""

        self._advance_internal_phase(tick_time)
        return self.internal_phase

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

    def compute_coherence_level(self, tick_time: float) -> float:
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
                from .. import tick_engine as te

                te.mark_for_update(self.id)
            return self.coherence
        weights = []
        complex_vecs = []
        for item in phases:
            if isinstance(item, (tuple, list)):
                ph = item[0]
                amp = item[1] if len(item) > 1 else 1.0
                created = item[2] if len(item) > 2 else Config.current_tick
                decay = getattr(Config, "tick_decay_factor", 1.0) ** (
                    max(0, Config.current_tick - created)
                )
            else:
                ph = item
                amp = 1.0
                decay = 1.0
            weight = amp * decay
            complex_vecs.append(weight * cmath.exp(1j * (ph % (2 * math.pi))))
            weights.append(weight)
        vector_sum = sum(complex_vecs)
        total_weight = sum(weights) if weights else 1.0
        self.coherence = abs(vector_sum) / total_weight
        threshold = self._coherence_threshold()
        if self.coherence > threshold:
            self.coherence_credit += self.coherence - threshold
        else:
            self.decoherence_debt += 0.1
        self._update_law_wave()
        if abs(self.coherence - old) > 0.05:
            from .. import tick_engine as te

            te.mark_for_update(self.id)
        self._coherence_cache[tick_time] = self.coherence
        return self.coherence

    def compute_decoherence_field(self, tick_time: float) -> float:
        if tick_time in self._decoherence_cache:
            return self._decoherence_cache[tick_time]
        phases = self.pending_superpositions.get(tick_time, [])
        old = self.decoherence
        if len(phases) < 2:
            self.decoherence = 0.0
            self.decoherence_debt = max(0.0, self.decoherence_debt - 0.1)
            if abs(self.decoherence - old) > 0.05:
                from .. import tick_engine as te

                te.mark_for_update(self.id)
            return self.decoherence
        norm = []
        weights = []
        for item in phases:
            if isinstance(item, (tuple, list)):
                ph = item[0]
                amp = item[1] if len(item) > 1 else 1.0
                created = item[2] if len(item) > 2 else Config.current_tick
                decay = getattr(Config, "tick_decay_factor", 1.0) ** (
                    max(0, Config.current_tick - created)
                )
            else:
                ph = item
                amp = 1.0
                decay = 1.0
            weight = amp * decay
            norm.append(ph % (2 * math.pi))
            weights.append(weight)
        total_weight = sum(weights) if weights else 1.0
        mean_phase = sum(w * p for w, p in zip(weights, norm)) / total_weight
        variance = (
            sum(w * (p - mean_phase) ** 2 for w, p in zip(weights, norm)) / total_weight
        )
        self.decoherence = variance
        if self.decoherence > 0.4:
            self.decoherence_debt += self.decoherence - 0.4
        else:
            self.coherence_credit += 0.05
        if abs(self.decoherence - old) > 0.05:
            from .. import tick_engine as te

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
            from .. import tick_engine as te

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
    ) -> tuple[bool, float | None]:
        """Attempt to merge near-aligned phases into a single tick."""
        tol = self._phase_drift_tolerance(tick_time)
        phases_only = [p[0] if isinstance(p, (tuple, list)) else p for p in raw_phases]
        diffs = [
            abs((p1 - p2 + math.pi) % (2 * math.pi) - math.pi)
            for i, p1 in enumerate(phases_only)
            for p2 in phases_only[i + 1 :]
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
        decoherence_strength: float,
        tick_time: int | None = None,
        graph: "CausalGraph" | None = None,
        threshold: float = 0.4,
        streak_required: int = 2,
    ) -> None:
        """Transition to the classical state when decoherence persists.

        Parameters
        ----------
        decoherence_strength:
            Current decoherence level for the node.
        tick_time:
            Global tick index for logging purposes.
        graph:
            The graph instance, used to emit law waves on collapse.
        threshold:
            Decoherence threshold that counts toward classicalization.
        streak_required:
            Number of consecutive ticks exceeding ``threshold`` required to
            trigger classicalization.
        """
        if decoherence_strength > threshold:
            self._decoherence_streak += 1
        else:
            self._decoherence_streak = 0

        if self._decoherence_streak >= streak_required and not self.is_classical:
            self.is_classical = True
            if tick_time is not None:
                record = {"tick": tick_time, "node": self.id, "event": "collapse_start"}
                self._log("collapse_front_log.json", record)
                if graph is not None:
                    graph.emit_law_wave(self.id, tick_time)
            self.update_node_type()

    def update_node_type(self) -> None:
        """Update :attr:`node_type` based on current state flags."""
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
            self._log("node_state_map.json", rec)
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
        self._log("tick_evaluation_log.json", record)
        if not fired:
            fail_rec = {
                "tick": tick_time,
                "node": self.id,
                "threshold": round(threshold, 4),
                "coherence": round(coherence, 4),
                "reason": reason or ("refractory" if refractory else "below_threshold"),
            }
            self._log("propagation_failure_log.json", fail_rec)

    def _log_tick_drop(
        self,
        tick_time: int,
        reason: str,
        *,
        tick_id: str | None = None,
        source: str | None = None,
        target: str | None = None,
        cumulative_delay: float | None = None,
        coherence: float | None = None,
    ) -> None:
        record = {
            "tick": tick_time,
            "node": self.id,
            "reason": reason,
            "coherence": round(
                coherence if coherence is not None else getattr(self, "coherence", 0.0),
                4,
            ),
            "node_type": self.node_type.value,
        }
        if tick_id is not None:
            record["tick_id"] = tick_id
        if source is not None:
            record["source"] = source
        if target is not None:
            record["target"] = target
        if cumulative_delay is not None:
            record["cumulative_delay"] = round(cumulative_delay, 4)
        self.tick_drop_counts[reason] += 1
        self._log("tick_drop_log.json", record)
        fail_rec = {
            "tick": tick_time,
            "node": self.id,
            "threshold": round(self.current_threshold, 4),
            "coherence": round(
                coherence if coherence is not None else getattr(self, "coherence", 0.0),
                4,
            ),
            "reason": reason,
        }
        self._log("propagation_failure_log.json", fail_rec)

    def schedule_tick(
        self,
        tick_time: float,
        incoming_phase: float,
        origin: str | None = None,
        created_tick: int | None = None,
        *,
        amplitude: float = 1.0,
        tick_id: str | None = None,
        cumulative_delay: float = 0.0,
        entangled_id: str | None = None,
    ) -> None:
        """Store an incoming phase for future evaluation.

        This method is thread-safe and may be called concurrently by
        propagation workers.

        Parameters
        ----------
        tick_time : float
            Global tick time at which the phase should be evaluated.
        incoming_phase : float
            Phase value being delivered.
        origin : str, optional
            ID of the node that emitted the phase.
        """

        if created_tick is None:
            created_tick = Config.current_tick

        record = (
            incoming_phase,
            amplitude,
            created_tick,
            cumulative_delay,
            tick_id,
            origin,
            entangled_id,
        )
        with self.lock:
            self.incoming_phase_queue[tick_time].append(record)
            self.incoming_tick_counts[tick_time] += 1
            self.pending_superpositions[tick_time].append(record)
            self._coherence_cache.pop(tick_time, None)
            self._decoherence_cache.pop(tick_time, None)
        print(
            f"[{self.id}] Received tick at {tick_time} with phase {incoming_phase:.2f}"
        )
        if origin is not None:
            self._log(
                "tick_delivery_log.json",
                {
                    "tick": tick_time,
                    "source": origin,
                    "node_id": self.id,
                    "stored_phase": incoming_phase,
                },
            )
        from .. import tick_engine as te

        te.mark_for_update(self.id)

    def should_tick(self, tick_time: float) -> tuple[bool, float | None, str]:
        """Return whether the node should fire at ``tick_time``."""

        from ..services.node_services import NodeTickDecisionService

        return NodeTickDecisionService(self, tick_time).decide()

    def get_phase_at(self, tick_time: float) -> float | None:
        return self._tick_phase_lookup.get(tick_time)

    def apply_tick(
        self,
        tick_time: float,
        phase: float,
        graph: "CausalGraph",
        origin: str = "self",
        *,
        entangled_id: str | None = None,
    ) -> None:
        """Emit a tick and propagate resulting phases to neighbours."""

        from ..services.node_services import NodeTickService

        NodeTickService(
            self,
            tick_time,
            phase,
            graph,
            origin,
            entangled_id=entangled_id,
        ).process()

    def _apply_suppression(self, tick_time: float) -> bool:
        items = list(self.incoming_phase_queue.get(tick_time, []))
        if not items:
            return False
        max_delay = getattr(Config, "max_cumulative_delay", 0)
        remaining = []
        for item in items:
            ph = item[0]
            amp = item[1] if len(item) > 1 else 1.0
            created = item[2] if len(item) > 2 else Config.current_tick
            delay = item[3] if len(item) > 3 else 0.0
            tid = item[4] if len(item) > 4 else None
            src = item[5] if len(item) > 5 else None
            ent_id = item[6] if len(item) > 6 else None
            if max_delay and delay > max_delay:
                self._log_tick_drop(
                    tick_time,
                    "event_horizon",
                    tick_id=tid,
                    source=src,
                    target=self.id,
                    cumulative_delay=delay,
                    coherence=self.compute_coherence_level(tick_time),
                )
                if tick_time in self.incoming_tick_counts:
                    self.incoming_tick_counts[tick_time] -= 1
                continue
            remaining.append((ph, amp, created, delay, tid, src, ent_id))

        if not remaining:
            self.incoming_phase_queue.pop(tick_time, None)
            self.pending_superpositions.pop(tick_time, None)
            return False

        self.incoming_phase_queue[tick_time] = remaining
        self.pending_superpositions[tick_time] = remaining
        coherence = self.compute_coherence_level(tick_time)
        min_coh = getattr(Config, "min_coherence_threshold", 0.0)
        if min_coh and coherence < min_coh:
            for ph, amp, created, delay, tid, src, ent_id in remaining:
                self._log_tick_drop(
                    tick_time,
                    "decoherence",
                    tick_id=tid,
                    source=src,
                    target=self.id,
                    cumulative_delay=delay,
                    coherence=coherence,
                )
                if tick_time in self.incoming_tick_counts:
                    self.incoming_tick_counts[tick_time] -= 1
            self.incoming_phase_queue.pop(tick_time, None)
            self.pending_superpositions.pop(tick_time, None)
            self._coherence_cache.pop(tick_time, None)
            self._decoherence_cache.pop(tick_time, None)
            return False
        return True

    def propagate_collapse(
        self,
        tick_time: int,
        graph: "CausalGraph",
        threshold: float = 0.5,
        depth: int = 1,
        visited: Set[str] | None = None,
    ) -> list[dict]:
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
            self._log("collapse_front_log.json", record)
        return chain

    def _log_collapse_chain(self, tick_time: int, collapsed: list[dict]) -> None:
        """Log collapse propagation events to file with depth info."""
        record = {
            "tick": tick_time,
            "source": self.id,
            "collapsed": collapsed,
            "collapsed_entity": self.id,
            "children_spawned": [c.get("node") for c in collapsed],
        }
        self._log("collapse_chain_log.json", record)

    def maybe_tick(self, global_tick: int, graph: "CausalGraph") -> None:
        """Evaluate queued phases and emit a tick if conditions are met."""
        if self.tick_history:
            from ..tick_engine.tick_router import TickRouter

            TickRouter.route_tick(self, self.tick_history[-1])

        tick_key = global_tick
        if tick_key not in self.incoming_phase_queue:
            for k in list(self.incoming_phase_queue.keys()):
                if math.isclose(k, global_tick, abs_tol=1e-6):
                    tick_key = k
                    break
        self._advance_internal_phase(tick_key)
        incoming = self.incoming_phase_queue.get(tick_key, [])
        if incoming:
            self.subjective_ticks += len(incoming)
        if tick_key in self.incoming_phase_queue:
            if not self._apply_suppression(tick_key):
                self.current_threshold = max(
                    self.base_threshold, self.current_threshold - 0.01
                )
                return
            should_fire, phase, reason = self.should_tick(tick_key)
            entangled_id = None
            for item in self.incoming_phase_queue.get(tick_key, []):
                if len(item) > 6 and item[6] is not None:
                    entangled_id = item[6]
                    break
            if should_fire and not self.is_classical:
                self.apply_tick(tick_key, phase, graph, entangled_id=entangled_id)
            else:
                drop_reason = "classical" if self.is_classical else reason
                self._log_tick_drop(tick_key, drop_reason)
                print(f"[{self.id}] {drop_reason} at {tick_key} cancelled tick")
            del self.incoming_phase_queue[tick_key]
            if tick_key in self.incoming_tick_counts:
                del self.incoming_tick_counts[tick_key]
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

    def _emit(self, tick_time: float) -> None:
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

    def __init__(
        self,
        source: str,
        target: str,
        attenuation: float,
        density: float,
        delay: int = 1,
        phase_shift: float = 0.0,
        weight: float = 1.0,
        *,
        density_specified: bool = True,
    ) -> None:
        self.source = source
        self.target = target
        self.attenuation = attenuation  # Multiplier for phase amplitude
        self.density = density  # Base density value
        self.density_specified = density_specified
        self.delay = delay
        self.phase_shift = phase_shift
        self.weight = weight
        self.tick_traffic = 0.0
        self._last_tick = 0

    def adjusted_delay(
        self,
        source_freq: float = 0.0,
        target_freq: float = 0.0,
        kappa: float = 1.0,
        *,
        graph: "CausalGraph | None" = None,
        radius: int | None = None,
    ) -> int:
        """Return delay modified by local density."""

        rho = self.density
        if graph is not None and (
            getattr(Config, "use_dynamic_density", False) or not self.density_specified
        ):
            rad = radius if radius is not None else getattr(Config, "density_radius", 1)
            rho = graph.compute_edge_density(self, radius=rad)

        adjusted = self.delay * (1 + kappa * rho) * self.weight
        delay_int = max(1, int(round(adjusted)))

        if getattr(Config, "log_verbosity", "info") == "debug" and graph is not None:
            from ..logging.logger import log_json

            log_json(
                "event",
                "delay_density_log",
                {
                    "edge": f"{self.source}->{self.target}",
                    "rho": rho,
                    "base": self.delay,
                    "result": delay_int,
                },
                tick=getattr(Config, "current_tick", 0),
            )

        return delay_int

    def propagate_phase(
        self,
        phase: float,
        global_tick: int,
        graph: "CausalGraph",
    ) -> None:
        target_node = graph.get_node(self.target)
        dt = global_tick - self._last_tick
        decay = getattr(Config, "traffic_decay", 0.9) ** max(1, dt)
        self.tick_traffic = self.tick_traffic * decay + 1
        self._last_tick = global_tick

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
        attenuated_amp = self.attenuation / self.weight
        scheduled_tick = global_tick + self.adjusted_delay(
            (
                graph.get_node(self.source).law_wave_frequency
                if hasattr(self, "source")
                else 0.0
            ),
            target_node.law_wave_frequency,
            getattr(Config, "delay_density_scaling", 1.0),
            graph=graph,
        )
        target_node.schedule_tick(
            scheduled_tick,
            shifted_phase,
            origin=self.source,
            created_tick=global_tick,
            amplitude=attenuated_amp,
        )
