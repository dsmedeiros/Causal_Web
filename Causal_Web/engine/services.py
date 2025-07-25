"""Service objects encapsulating large behaviour blocks."""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass

from ..config import Config
from .logger import log_json
from .tick import GLOBAL_TICK_POOL
from .node import Node, NodeType


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
    def _register_tick(self):
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
            {"node_id": n.id, "tick_time": self.tick_time, "phase": self.phase},
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
    def _propagate_edges(self):
        n = self.node
        g = self.graph
        if self.origin != "self" and any(
            e.target == self.origin for e in g.get_edges_from(n.id)
        ):
            log_json(
                Config.output_path("refraction_log.json"),
                {"tick": self.tick_time, "recursion_from": self.origin, "node": n.id},
            )
        from .tick_engine import kappa

        for edge in n._fanout_edges(g):
            target = g.get_node(edge.target)
            delay = edge.adjusted_delay(
                n.law_wave_frequency, target.law_wave_frequency, kappa
            )
            attenuated = self.phase * edge.attenuation
            shifted = attenuated + edge.phase_shift
            log_json(
                Config.output_path("tick_propagation_log.json"),
                {
                    "source": n.id,
                    "target": target.id,
                    "tick_time": self.tick_time,
                    "arrival_time": self.tick_time + delay,
                    "phase": shifted,
                },
            )
            if target.node_type == NodeType.DECOHERENT:
                alts = g.get_edges_from(target.id)
                if alts:
                    alt = alts[0]
                    alt_tgt = g.get_node(alt.target)
                    alt_delay = alt.adjusted_delay(
                        target.law_wave_frequency, alt_tgt.law_wave_frequency, kappa
                    )
                    alt_tgt.schedule_tick(
                        self.tick_time + delay + alt_delay,
                        shifted,
                        origin=n.id,
                        created_tick=self.tick_time,
                    )
                    target.node_type = NodeType.REFRACTIVE
                    log_json(
                        Config.output_path("refraction_log.json"),
                        {
                            "tick": self.tick_time,
                            "from": n.id,
                            "via": target.id,
                            "to": alt_tgt.id,
                        },
                    )
                    continue
            target.schedule_tick(
                self.tick_time + delay,
                shifted,
                origin=n.id,
                created_tick=self.tick_time,
            )


class GraphLoadService:
    """Populate a :class:`Graph` from a JSON file."""

    def __init__(self, graph, path: str):
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
    def _load_nodes(self, nodes_data):
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
                    "refractory_period", getattr(Config, "refractory_period", 2.0)
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
    def _load_edges(self, edges_data):
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
    def _load_bridges(self, bridges):
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
    def _load_meta_nodes(self, meta_nodes):
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
