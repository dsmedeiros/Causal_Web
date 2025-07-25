"""Service objects encapsulating large behaviour blocks."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Any
from concurrent.futures import ThreadPoolExecutor
import math
import cmath
import numpy as np

from ..config import Config
from .logger import log_json
from .tick import Tick, GLOBAL_TICK_POOL
from .node import Node, NodeType, Edge
from .graph import CausalGraph
from . import tick_engine as te


@dataclass
class NodeTickService:
    """Lifecycle manager for :meth:`Node.apply_tick`."""

    node: Node
    tick_time: int
    phase: float
    graph: any
    origin: str = "self"

    def process(self) -> None:
        if not self._pre_check():
            return
        self._register_tick()
        self._propagate_edges()
        if self.origin == "self":
            collapsed = self.node.propagate_collapse(self.tick_time, self.graph)
            if collapsed:
                self.node._log_collapse_chain(self.tick_time, collapsed)

    # ------------------------------------------------------------------
    def _pre_check(self) -> bool:
        from . import tick_engine as te

        if self.node.node_type == NodeType.NULL:
            log_json(
                Config.output_path("boundary_interaction_log.json"),
                {
                    "tick": self.tick_time,
                    "void": self.node.id,
                    "origin": self.origin,
                },
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
                {
                    "tick": self.tick_time,
                    "node": self.node.id,
                    "origin": self.origin,
                },
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
        n.current_tick += 1
        n.subjective_ticks += 1
        n.last_tick_time = self.tick_time
        n.current_threshold = min(n.current_threshold + 0.05, 1.0)
        n.phase = self.phase
        n.tick_history.append(tick_obj)
        log_json(
            Config.output_path("tick_emission_log.json"),
            {
                "node_id": n.id,
                "tick_time": self.tick_time,
                "phase": self.phase,
            },
        )
        if self.origin == "self":
            n.emitted_tick_times.add(self.tick_time)
        else:
            n.received_tick_times.add(self.tick_time)
        n._tick_phase_lookup[self.tick_time] = self.phase
        from .tick_router import TickRouter

        TickRouter.route_tick(n, tick_obj)
        n.collapse_origin[self.tick_time] = self.origin
        print(
            f"[{n.id}] Tick at {self.tick_time} via {self.origin.upper()} | Phase: {self.phase:.2f}"
        )
        n._update_memory(self.tick_time, self.origin)
        n._adapt_behavior()
        n.update_node_type()
        return tick_obj

    # ------------------------------------------------------------------
    def _propagate_edges(self) -> None:
        EdgePropagationService(
            node=self.node,
            tick_time=self.tick_time,
            phase=self.phase,
            origin=self.origin,
            graph=self.graph,
        ).propagate()


@dataclass
class EdgePropagationService:
    """Handle tick propagation across outgoing edges."""

    node: Node
    tick_time: int
    phase: float
    origin: str
    graph: Any

    def propagate(self) -> None:
        self._log_recursion()
        from .tick_engine import kappa

        for edge in self.node._fanout_edges(self.graph):
            self._propagate_edge(edge, kappa)

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
            self.node.law_wave_frequency, target.law_wave_frequency, kappa
        )
        shifted = self._shift_phase(edge)
        self._log_propagation(target, delay, shifted)
        if self._handle_refraction(target, delay, shifted, kappa):
            return
        target.schedule_tick(
            self.tick_time + delay,
            shifted,
            origin=self.node.id,
            created_tick=self.tick_time,
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
class NodeMetricsResultService:
    """Process per-node metric records."""

    graph: Any
    results: list
    tick: int
    last_coherence: dict

    def process(self) -> dict:
        self._init_logs()
        for rec in self.results:
            self._handle_record(*rec)
        return self.logs

    # ------------------------------------------------------------------
    def _init_logs(self) -> None:
        self.logs = {
            "decoherence_log": {},
            "coherence_log": {},
            "classical_state": {},
            "coherence_velocity": {},
            "law_wave_log": {},
            "stable_frequency_log": {},
            "interference_log": {},
            "credit_log": {},
            "debt_log": {},
            "type_log": {},
        }

    # ------------------------------------------------------------------
    def _handle_record(
        self,
        node_id: str,
        deco: float,
        coh: float,
        inter: int,
        ntype: int,
        credit: float,
        debt: float,
    ) -> None:
        node = self.graph.get_node(node_id)
        prev = self.last_coherence.get(node_id, coh)
        delta = coh - prev
        self.last_coherence[node_id] = coh
        node.coherence_velocity = delta
        node.update_classical_state(deco, tick_time=self.tick, graph=self.graph)
        self._update_stability(node_id, node)
        self._record_logs(node_id, deco, coh, inter, ntype, credit, debt, delta, node)

    # ------------------------------------------------------------------
    def _update_stability(self, node_id: str, node: Node) -> None:
        record = te._law_wave_stability.setdefault(node_id, {"freqs": [], "stable": 0})
        record["freqs"].append(node.law_wave_frequency)
        if len(record["freqs"]) > 5:
            record["freqs"].pop(0)
        if len(record["freqs"]) == 5:
            if np.std(record["freqs"]) < 0.01:
                record["stable"] += 1
            else:
                record["stable"] = 0
        if record["stable"] >= 10:
            node.refractory_period = max(1.0, node.refractory_period - 0.1)
            log_json(
                Config.output_path("law_drift_log.json"),
                {
                    "tick": self.tick,
                    "node": node_id,
                    "new_refractory_period": node.refractory_period,
                },
            )
            record["stable"] = 0
        if record["stable"] >= 5:
            self.logs["stable_frequency_log"][node_id] = round(
                np.mean(record["freqs"]), 4
            )

    # ------------------------------------------------------------------
    def _record_logs(
        self,
        node_id: str,
        deco: float,
        coh: float,
        inter: int,
        ntype: int,
        credit: float,
        debt: float,
        delta: float,
        node: Node,
    ) -> None:
        self.logs["decoherence_log"][node_id] = round(deco, 4)
        self.logs["coherence_log"][node_id] = round(coh, 4)
        self.logs["classical_state"][node_id] = getattr(node, "is_classical", False)
        self.logs["coherence_velocity"][node_id] = round(delta, 5)
        self.logs["law_wave_log"][node_id] = round(node.law_wave_frequency, 4)
        self.logs["interference_log"][node_id] = inter
        self.logs["credit_log"][node_id] = round(credit, 3)
        self.logs["debt_log"][node_id] = round(debt, 3)
        self.logs["type_log"][node_id] = ntype


@dataclass
class NodeMetricsService:
    """Collect and persist per-tick node metrics."""

    graph: any
    last_coherence: dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    def log_metrics(self, tick: int) -> None:
        """Gather metrics for ``tick`` and write log files."""
        results = self._gather(tick)
        logs = self._process_results(results, tick)
        self._maybe_handle_clusters(tick)
        if tick % getattr(Config, "log_interval", 1) == 0:
            self._write_logs(tick, logs)

    # ------------------------------------------------------------------
    def _gather(self, tick: int) -> list:
        with ThreadPoolExecutor(
            max_workers=getattr(Config, "thread_count", None)
        ) as ex:
            return list(
                ex.map(lambda n: self._compute(n, tick), self.graph.nodes.values())
            )

    # ------------------------------------------------------------------
    @staticmethod
    def _compute(
        node: Node, tick: int
    ) -> tuple[str, float, float, int, str, float, float]:
        decoherence = node.compute_decoherence_field(tick)
        coherence = node.compute_coherence_level(tick)
        interference = len(node.pending_superpositions.get(tick, []))
        return (
            node.id,
            decoherence,
            coherence,
            interference,
            node.node_type.value,
            node.coherence_credit,
            node.decoherence_debt,
        )

    # ------------------------------------------------------------------
    def _process_results(self, results: list, tick: int) -> dict:
        return NodeMetricsResultService(
            graph=self.graph,
            results=results,
            tick=tick,
            last_coherence=self.last_coherence,
        ).process()

    # ------------------------------------------------------------------
    def _maybe_handle_clusters(self, tick: int) -> None:
        if tick % getattr(Config, "cluster_interval", 1) != 0:
            return
        clusters = self.graph.hierarchical_clusters()
        self.graph.create_meta_nodes(clusters.get(0, []))
        if tick % getattr(Config, "log_interval", 1) == 0:
            log_json(Config.output_path("cluster_log.json"), {str(tick): clusters})

    # ------------------------------------------------------------------
    def _write_logs(self, tick: int, logs: dict) -> None:
        log_json(
            Config.output_path("law_wave_log.json"),
            {str(tick): logs["law_wave_log"]},
        )
        if logs["stable_frequency_log"]:
            log_json(
                Config.output_path("stable_frequency_log.json"),
                {str(tick): logs["stable_frequency_log"]},
            )
        log_json(
            Config.output_path("decoherence_log.json"),
            {str(tick): logs["decoherence_log"]},
        )
        log_json(
            Config.output_path("coherence_log.json"),
            {str(tick): logs["coherence_log"]},
        )
        log_json(
            Config.output_path("coherence_velocity_log.json"),
            {str(tick): logs["coherence_velocity"]},
        )
        log_json(
            Config.output_path("classicalization_map.json"),
            {str(tick): logs["classical_state"]},
        )
        log_json(
            Config.output_path("interference_log.json"),
            {str(tick): logs["interference_log"]},
        )
        log_json(
            Config.output_path("tick_density_map.json"),
            {str(tick): logs["interference_log"]},
        )
        log_json(
            Config.output_path("node_state_log.json"),
            {
                str(tick): {
                    "type": logs["type_log"],
                    "credit": logs["credit_log"],
                    "debt": logs["debt_log"],
                }
            },
        )


class GraphLoadService:
    """Populate a :class:`Graph` from a JSON file."""

    def __init__(self, graph: "CausalGraph", path: str) -> None:
        self.graph = graph
        self.path = path

    # ------------------------------------------------------------------
    def load(self) -> None:
        with open(self.path, "r") as f:
            data = json.load(f)
        self._reset()
        self._load_nodes(data.get("nodes", []))
        self._load_edges(data.get("edges", []))
        self._load_bridges(data.get("bridges", []))
        self.graph.tick_sources.extend(data.get("tick_sources", []))
        self._load_meta_nodes(data.get("meta_nodes", {}))
        self.graph.identify_boundaries()

    # ------------------------------------------------------------------
    def _reset(self) -> None:
        g = self.graph
        g.nodes.clear()
        g.edges.clear()
        g.edges_from.clear()
        g.edges_to.clear()
        g.bridges.clear()
        g.bridges_by_node.clear()
        g.tick_sources = []
        g.spatial_index.clear()
        g.meta_nodes.clear()

    # ------------------------------------------------------------------
    def _load_nodes(self, nodes_data: list | dict) -> None:
        g = self.graph
        if isinstance(nodes_data, dict):
            nodes_iter = [dict(v, id=k) for k, v in nodes_data.items()]
        else:
            nodes_iter = nodes_data
        for node_data in nodes_iter:
            node_id = node_data.get("id")
            if node_id is None:
                continue
            g.add_node(
                node_id,
                x=node_data.get("x", 0.0),
                y=node_data.get("y", 0.0),
                frequency=node_data.get("frequency", 1.0),
                refractory_period=node_data.get(
                    "refractory_period",
                    getattr(Config, "refractory_period", 2.0),
                ),
                base_threshold=node_data.get("base_threshold", 0.5),
                phase=node_data.get("phase", 0.0),
                origin_type=node_data.get("origin_type", "seed"),
                generation_tick=node_data.get("generation_tick", 0),
                parent_ids=node_data.get("parent_ids"),
            )
            goals = node_data.get("goals")
            if goals is not None:
                g.nodes[node_id].goals = goals

    # ------------------------------------------------------------------
    def _load_edges(self, edges_data: list | dict) -> None:
        g = self.graph
        if isinstance(edges_data, dict):
            edges_iter = []
            for src, targets in edges_data.items():
                if isinstance(targets, dict):
                    for tgt, params in targets.items():
                        rec = {"from": src, "to": tgt}
                        if isinstance(params, dict):
                            rec.update(params)
                        edges_iter.append(rec)
                else:
                    for tgt in targets:
                        if isinstance(tgt, str):
                            edges_iter.append({"from": src, "to": tgt})
                        elif isinstance(tgt, dict):
                            rec = {"from": src}
                            rec.update(tgt)
                            edges_iter.append(rec)
        else:
            edges_iter = edges_data

        for edge in edges_iter:
            src = edge.get("from")
            tgt = edge.get("to")
            if src is None or tgt is None:
                continue
            if src == tgt:
                g.tick_sources.append(
                    {
                        "node_id": src,
                        "tick_interval": edge.get("delay", 1),
                        "phase": edge.get("phase_shift", 0.0),
                    }
                )
                continue
            g.add_edge(
                src,
                tgt,
                attenuation=edge.get("attenuation", 1.0),
                density=edge.get("density", 0.0),
                delay=edge.get("delay", 1),
                phase_shift=edge.get("phase_shift", 0.0),
                weight=edge.get("weight"),
            )

    # ------------------------------------------------------------------
    def _load_bridges(self, bridges: list[dict]) -> None:
        g = self.graph
        for bridge in bridges:
            src = bridge.get("from")
            tgt = bridge.get("to")
            if src is None or tgt is None:
                continue
            g.add_bridge(
                src,
                tgt,
                bridge_type=bridge.get("bridge_type", "braided"),
                phase_offset=bridge.get("phase_offset", 0.0),
                drift_tolerance=bridge.get("drift_tolerance"),
                decoherence_limit=bridge.get("decoherence_limit"),
                initial_strength=bridge.get("initial_strength", 1.0),
                medium_type=bridge.get("medium_type", "standard"),
                mutable=bridge.get("mutable", True),
                seeded=True,
                formed_at_tick=0,
            )

    # ------------------------------------------------------------------
    def _load_meta_nodes(self, meta_nodes: dict) -> None:
        g = self.graph
        from .meta_node import MetaNode

        for mid, meta in meta_nodes.items():
            members = list(meta.get("members", []))
            constraints = dict(meta.get("constraints", {}))
            mtype = meta.get("type", "Configured")
            origin = meta.get("origin")
            collapsed = bool(meta.get("collapsed", False))
            x = float(meta.get("x", 0.0))
            y = float(meta.get("y", 0.0))
            g.meta_nodes[mid] = MetaNode(
                members,
                g,
                meta_type=mtype,
                constraints=constraints,
                origin=origin,
                collapsed=collapsed,
                x=x,
                y=y,
                id=mid,
            )


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
            if isinstance(item, (tuple, list)) and len(item) == 2:
                ph, created = item
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
        self,
        coherence: float,
        refractory: bool,
        fired: bool,
        reason: str | None = None,
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
            {
                "tick": self.tick_time,
                "node": self.node.id,
                "reason": "below_count",
            },
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
            {
                "tick": self.tick_time,
                "node": self.node.id,
                "reason": "threshold",
            },
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


@dataclass
class BridgeApplyService:
    """Lifecycle handler for :meth:`Bridge.apply`."""

    bridge: Any
    tick_time: int
    graph: Any

    def process(self) -> None:
        self._gather_nodes()
        self.bridge.decay(self.tick_time)
        self.bridge.try_reform(self.tick_time, self.node_a, self.node_b)
        if not self._validate_collapse():
            return
        if self._check_drift():
            return
        if self._handle_decoherence():
            return
        if not self.bridge.active:
            self.bridge.update_state(self.tick_time)
            return
        self._propagate()
        self._record_metrics()
        self.bridge.update_state(self.tick_time)

    # ------------------------------------------------------------------
    def _gather_nodes(self) -> None:
        self.node_a = self.graph.get_node(self.bridge.node_a_id)
        self.node_b = self.graph.get_node(self.bridge.node_b_id)
        self.phase_a = self.node_a.get_phase_at(self.tick_time)
        self.phase_b = self.node_b.get_phase_at(self.tick_time)
        self.a_collapsed = self.node_a.collapse_origin.get(self.tick_time) == "self"
        self.b_collapsed = self.node_b.collapse_origin.get(self.tick_time) == "self"

    # ------------------------------------------------------------------
    def _validate_collapse(self) -> bool:
        return self.a_collapsed != self.b_collapsed

    # ------------------------------------------------------------------
    def _check_drift(self) -> bool:
        if (
            self.bridge.drift_tolerance is not None
            and self.phase_a is not None
            and self.phase_b is not None
        ):
            drift = abs(
                (self.phase_a - self.phase_b + math.pi) % (2 * math.pi) - math.pi
            )
            if drift > self.bridge.drift_tolerance:
                print(
                    f"[BRIDGE] Drift too high at tick {self.tick_time}: {drift:.2f} > {self.bridge.drift_tolerance}"
                )
                self.bridge._log_event(self.tick_time, "bridge_drift", drift)
                self.bridge.trust_score = max(0.0, self.bridge.trust_score - 0.05)
                self.bridge.reinforcement_streak = 0
                return True
        return False

    # ------------------------------------------------------------------
    def _handle_decoherence(self) -> bool:
        deco_a = self.node_a.compute_decoherence_field(self.tick_time)
        deco_b = self.node_b.compute_decoherence_field(self.tick_time)
        avg = (deco_a + deco_b) / 2
        debt = (self.node_a.decoherence_debt + self.node_b.decoherence_debt) / 6.0
        rupture_chance = avg + debt
        self.bridge.decoherence_exposure.append(avg)
        if len(self.bridge.decoherence_exposure) > 20:
            self.bridge.decoherence_exposure.pop(0)
        if self.bridge.probabilistic_bridge_failure(rupture_chance):
            self._record_rupture(avg, "decoherence")
            return True
        if self.bridge.decoherence_limit and self.bridge.last_activation is not None:
            if (
                self.tick_time - self.bridge.last_activation
                > self.bridge.decoherence_limit
            ):
                print(f"[BRIDGE] Decohered at tick {self.tick_time}, disabling bridge.")
                self.bridge.active = False
                self._record_rupture(avg, "decoherence_limit")
                return True
        return False

    # ------------------------------------------------------------------
    def _record_rupture(self, avg_decoherence: float, reason: str) -> None:
        self.bridge.last_rupture_tick = self.tick_time
        self.bridge._log_event(self.tick_time, "bridge_ruptured", avg_decoherence)
        avg_coh = (
            self.node_a.compute_coherence_level(self.tick_time)
            + self.node_b.compute_coherence_level(self.tick_time)
        ) / 2
        self.bridge._log_rupture(self.tick_time, reason, avg_coh)
        self.bridge._log_dynamics(
            self.tick_time, "ruptured", {"decoherence": avg_decoherence}
        )
        self.bridge.rupture_history.append((self.tick_time, avg_decoherence))
        self.bridge.trust_score = max(0.0, self.bridge.trust_score - 0.1)
        self.bridge.reinforcement_streak = 0
        from . import tick_engine as te

        te.trigger_csp(
            self.bridge.bridge_id, self.node_a.x, self.node_a.y, self.tick_time
        )
        self.bridge.update_state(self.tick_time)

    # ------------------------------------------------------------------
    def _propagate(self) -> None:
        if self.bridge.bridge_type == "braided":
            if self.a_collapsed:
                phase = self.node_a.get_phase_at(self.tick_time)
                self.node_b.apply_tick(
                    self.tick_time,
                    phase + self.bridge.phase_offset,
                    self.graph,
                    origin="bridge",
                )
                self.node_a.entangled_with.add(self.node_b.id)
                self.node_b.entangled_with.add(self.node_a.id)
            elif self.b_collapsed:
                phase = self.node_b.get_phase_at(self.tick_time)
                self.node_a.apply_tick(
                    self.tick_time,
                    phase + self.bridge.phase_offset,
                    self.graph,
                    origin="bridge",
                )
                self.node_a.entangled_with.add(self.node_b.id)
                self.node_b.entangled_with.add(self.node_a.id)
        elif self.bridge.bridge_type in {"unidirectional", "mirror"}:
            if self.a_collapsed:
                phase = self.node_a.get_phase_at(self.tick_time)
                self.node_b.apply_tick(
                    self.tick_time,
                    phase + self.bridge.phase_offset,
                    self.graph,
                    origin="bridge",
                )
                self.node_a.entangled_with.add(self.node_b.id)
                self.node_b.entangled_with.add(self.node_a.id)

    # ------------------------------------------------------------------
    def _record_metrics(self) -> None:
        if self.bridge.active:
            self.bridge.tick_load += 1
            if self.phase_a is not None and self.phase_b is not None:
                self.bridge.phase_drift += abs(
                    (self.phase_a - self.phase_b + math.pi) % (2 * math.pi) - math.pi
                )
            self.bridge.coherence_flux += (
                self.node_a.coherence + self.node_b.coherence
            ) / 2
        self.bridge.last_activation = self.tick_time
        if self.bridge.active:
            self.bridge.last_active_tick = self.tick_time
        self.bridge.reinforcement_streak += 1
        self.bridge.trust_score = min(1.0, self.bridge.trust_score + 0.01)


@dataclass
class GlobalDiagnosticsService:
    """Compute run-level diagnostic metrics."""

    graph: Any

    def export(self) -> None:
        deco_lines = self._load_log("decoherence_log.json")
        if not deco_lines:
            return
        entropy_delta, stability = self._entropy_metrics(deco_lines)
        resilience = self._resilience_index()
        adaptivity = self._adaptivity_index()
        diagnostics = {
            "coherence_stability_score": stability,
            "entropy_delta": round(entropy_delta, 3),
            "collapse_resilience_index": resilience,
            "network_adaptivity_index": round(adaptivity, 3),
        }
        with open(Config.output_path("global_diagnostics.json"), "w") as f:
            json.dump(diagnostics, f, indent=2)
        print("✅ Global diagnostics exported")

    # ------------------------------------------------------------------
    def _load_log(self, name: str) -> list:
        lines = []
        try:
            with open(Config.output_path(name)) as f:
                for line in f:
                    lines.append(json.loads(line))
        except FileNotFoundError:
            pass
        return lines

    # ------------------------------------------------------------------
    @staticmethod
    def _entropy_metrics(entries: list) -> tuple[float, float]:
        first = next(iter(entries[0].values()))
        last = next(iter(entries[-1].values()))
        entropy_first = sum(first.values())
        entropy_last = sum(last.values())
        entropy_delta = entropy_last - entropy_first
        stability = []
        for entry in entries:
            vals = list(entry.values())[0]
            stability.append(sum(1 for v in vals.values() if v < 0.4) / len(vals))
        coherence_stability_score = round(sum(stability) / len(stability), 3)
        return entropy_delta, coherence_stability_score

    # ------------------------------------------------------------------
    @staticmethod
    def _resilience_index() -> float:
        collapse_events = 0
        try:
            with open(Config.output_path("classicalization_map.json")) as f:
                for line in f:
                    states = json.loads(line)
                    collapse_events += sum(
                        1 for v in next(iter(states.values())).values() if v
                    )
        except FileNotFoundError:
            pass
        resilience = 0
        try:
            with open(Config.output_path("law_wave_log.json")) as f:
                resilience = sum(1 for _ in f)
        except FileNotFoundError:
            pass
        return round(resilience / (collapse_events or 1), 3)

    # ------------------------------------------------------------------
    def _adaptivity_index(self) -> float:
        if not self.graph.bridges:
            return 0.0
        active = sum(1 for b in self.graph.bridges if b.active)
        return active / len(self.graph.bridges)
