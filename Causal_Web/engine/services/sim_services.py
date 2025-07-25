"""Simulation-level service classes from the old services module."""

from __future__ import annotations

import json
import math
import cmath
from dataclasses import dataclass, field
from typing import Any
from concurrent.futures import ThreadPoolExecutor
import numpy as np

from ...config import Config
from ..logger import log_json
from ..node import Node, NodeType, Edge
from ..tick import GLOBAL_TICK_POOL  # only for typing, no direct use
from .. import tick_engine as te


@dataclass
class NodeMetricsResultService:
    """Process per-node metric records."""

    graph: Any
    results: list
    tick: int
    last_coherence: dict

    def process(self) -> dict:
        """Aggregate node metric results into structured logs."""

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

    graph: Any
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
            Config.output_path("law_wave_log.json"), {str(tick): logs["law_wave_log"]}
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
            Config.output_path("coherence_log.json"), {str(tick): logs["coherence_log"]}
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


@dataclass
class GraphLoadService:
    """Populate a :class:`Graph` from a JSON file."""

    def __init__(self, graph: "CausalGraph", path: str) -> None:
        """Create a loader for ``graph`` using ``path`` as input."""

        self.graph = graph
        self.path = path

    # ------------------------------------------------------------------
    def load(self) -> None:
        """Read ``path`` and populate ``graph`` with its contents."""

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
                self.graph.tick_sources.append(
                    {
                        "node_id": src,
                        "tick_interval": edge.get("delay", 1),
                        "phase": edge.get("phase_shift", 0.0),
                    }
                )
                continue
            self.graph.add_edge(
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
        from ..meta_node import MetaNode

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
class BridgeApplyService:
    """Lifecycle handler for :meth:`Bridge.apply`."""

    bridge: Any
    tick_time: int
    graph: Any

    def process(self) -> None:
        """Run the bridge activation sequence for ``tick_time``."""

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
        """Write overall diagnostics derived from simulation logs."""

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
