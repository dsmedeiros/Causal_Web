"""Node evaluation and propagation helpers."""

from __future__ import annotations

import math
import json
from collections import deque

import numpy as np

from ...config import Config
from ..models.graph import CausalGraph
from ..models.node import Node
from ..logging.logger import log_json, log_manager
from ..models.logging import (
    NodeEmergenceLog,
    NodeEmergencePayload,
    StructuralGrowthLog,
    StructuralGrowthPayload,
)

# The global graph instance is injected at runtime from :mod:`core`
graph: CausalGraph | None = None


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

nodes_to_update: set[str] = set()


def attach_graph(g: CausalGraph) -> None:
    """Attach the global graph instance used by this module."""

    global graph
    graph = g


def mark_for_update(node_id: str) -> None:
    """Flag ``node_id`` for evaluation on the next tick."""

    nodes_to_update.add(node_id)


# ---------------------------------------------------------------------------
# Firing limits
# ---------------------------------------------------------------------------

_current_firing_count = 0
_cluster_firing_counts: dict[int, int] = {}


def reset_firing_limits() -> None:
    """Reset per-tick firing counters."""

    global _current_firing_count, _cluster_firing_counts
    _current_firing_count = 0
    _cluster_firing_counts = {}


def register_firing(node: Node) -> bool:
    """Register ``node`` firing and enforce concurrency limits."""

    global _current_firing_count, _cluster_firing_counts
    total_limit = getattr(Config, "total_max_concurrent_firings", 0)
    cluster_limit = getattr(Config, "max_concurrent_firings_per_cluster", 0)

    cluster = node.cluster_ids.get(0)
    if total_limit and _current_firing_count >= total_limit:
        return False
    if cluster_limit and cluster is not None:
        if _cluster_firing_counts.get(cluster, 0) >= cluster_limit:
            return False

    _current_firing_count += 1
    if cluster is not None:
        _cluster_firing_counts[cluster] = _cluster_firing_counts.get(cluster, 0) + 1
    return True


# ---------------------------------------------------------------------------
# Coherence forcing
# ---------------------------------------------------------------------------

_recent_node_counts: deque[int] = deque(maxlen=500)
_dynamic_coherence_offset: float = 0.0


def apply_global_forcing(tick: int) -> None:
    """Apply rhythmic modulation to all nodes with a startup ramp."""

    assert graph is not None
    jitter = Config.phase_jitter
    wave = Config.coherence_wave
    ramp = min(tick / max(1, Config.forcing_ramp_ticks), 1.0)
    for node in graph.nodes.values():
        if jitter["amplitude"] and jitter.get("period", 0):
            node.phase += (
                ramp
                * jitter["amplitude"]
                * math.sin(2 * math.pi * tick / jitter["period"])
            )
        if wave["amplitude"] and wave.get("period", 0):
            mod = (
                ramp * wave["amplitude"] * math.sin(2 * math.pi * tick / wave["period"])
            )
            node.current_threshold = max(0.1, node.current_threshold - mod)
        mark_for_update(node.id)


def update_coherence_constraints() -> None:
    """Adjust node thresholds based on network size and growth."""

    assert graph is not None
    global _recent_node_counts, _dynamic_coherence_offset
    count = len(graph.nodes)
    _recent_node_counts.append(count)
    if len(_recent_node_counts) > 5:
        _recent_node_counts.popleft()
    growth = 0
    if len(_recent_node_counts) >= 2:
        growth = _recent_node_counts[-1] - _recent_node_counts[0]
    offset = min(0.3, 0.01 * count + 0.02 * growth)
    _dynamic_coherence_offset = offset
    for node in graph.nodes.values():
        node.dynamic_offset = offset
        base = max(1.0, 2 + offset * 5)
        ramp = getattr(Config, "FORMATION_REFRACTORY_RAMP", 20)
        progress = min(Config.current_tick, ramp) / ramp
        scale = 0.5 + 0.5 * progress
        node.refractory_period = max(0.5, base * scale)
        mark_for_update(node.id)


# ---------------------------------------------------------------------------
# Node evaluation
# ---------------------------------------------------------------------------


def evaluate_nodes(global_tick: int) -> None:
    """Evaluate only nodes flagged in :data:`nodes_to_update`."""

    assert graph is not None
    reset_firing_limits()
    for node_id in list(nodes_to_update):
        node = graph.get_node(node_id)
        if not node:
            nodes_to_update.discard(node_id)
            continue
        node.maybe_tick(global_tick, graph)
        if not node.incoming_phase_queue:
            nodes_to_update.discard(node_id)


# ---------------------------------------------------------------------------
# Propagation services
# ---------------------------------------------------------------------------

_sip_pending: list[tuple[str, list[str], int, str]] = []
_csp_seeds: list[dict] = []
_sip_success_count = 0
_sip_failure_count = 0
_csp_success_count = 0
_csp_failure_count = 0
_spawn_counts: dict[str, int] = {}
_spawn_tick = -1


class CSPSeedService:
    """Process collapse-seeded propagation events."""

    def __init__(self, graph: CausalGraph):
        self.graph = graph
        self.success = 0
        self.failure = 0

    def process(self, seeds: list, tick: int) -> list:
        remaining = []
        for seed in list(seeds):
            if tick - seed["tick"] < Config.CSP_STABILIZATION_WINDOW:
                remaining.append(seed)
                continue
            if np.random.rand() > 0.5:
                if (
                    _spawn_counts.get(seed["parent"], 0)
                    >= Config.max_children_per_node
                    > 0
                ):
                    if Config.is_log_enabled("event", "propagation_failure_log"):
                        log_json(
                            "event",
                            "propagation_failure_log",
                            {
                                "type": "SPAWN_LIMIT",
                                "parent": seed["parent"],
                                "origin_type": "CSP",
                            },
                            tick=tick,
                        )
                    continue
                self._spawn_node(seed, tick)
            else:
                self._record_failure(seed, tick)
        return remaining

    def _spawn_node(self, seed: dict, tick: int) -> None:
        node_id = seed["id"].replace("CSPseed", "CSP")
        self.graph.add_node(
            node_id,
            x=seed["x"],
            y=seed["y"],
            frequency=np.random.uniform(0.2, 2.0),
            origin_type="CSP",
            generation_tick=tick,
            parent_ids=[seed["parent"]],
        )
        parent_id = seed["parent"]
        if "->" in parent_id:
            src, tgt = parent_id.split("->", 1)
            for pid in (src, tgt):
                if pid in self.graph.nodes:
                    self.graph.add_edge(pid, node_id)
        else:
            self.graph.add_edge(parent_id, node_id)
        payload = NodeEmergencePayload(
            node_id=node_id,
            origin_type="CSP",
            parents=[seed["parent"]],
        )
        entry = NodeEmergenceLog(tick=tick, value=payload)
        log_json("event", "node_emergence_log", entry.model_dump(), tick=tick)
        log_json(
            "event",
            "collapse_chain_log",
            {"source": seed["parent"], "children_spawned": [node_id]},
            tick=tick,
        )
        self.success += 1
        _spawn_counts[seed["parent"]] = _spawn_counts.get(seed["parent"], 0) + 1

    def _record_failure(self, seed: dict, tick: int) -> None:
        parent = self.graph.get_node(seed["parent"])
        if parent:
            parent.decoherence_debt += Config.CSP_ENTROPY_INJECTION
        log_json(
            "event",
            "propagation_failure_log",
            {
                "type": "CSP_FAILURE",
                "parent": seed["parent"],
                "reason": "Seed failed to cohere",
                "entropy_injected": Config.CSP_ENTROPY_INJECTION,
                "origin_type": "CSP",
                "location": [seed["x"], seed["y"]],
            },
            tick=tick,
        )
        self.failure += 1


class SIPRecombinationService:
    """Handle SIP recombination child spawning."""

    def __init__(self, graph: CausalGraph):
        self.graph = graph

    @staticmethod
    def _limit_reached(a_id: str, b_id: str) -> bool:
        return any(
            _spawn_counts.get(pid, 0) >= Config.max_children_per_node > 0
            for pid in (a_id, b_id)
        )

    @staticmethod
    def _log_limit(parent_a, parent_b, tick: int) -> None:
        if Config.is_log_enabled("event", "propagation_failure_log"):
            log_json(
                "event",
                "propagation_failure_log",
                {
                    "type": "SPAWN_LIMIT",
                    "parent": f"{parent_a.id},{parent_b.id}",
                    "origin_type": "SIP_RECOMB",
                },
                tick=tick,
            )

    @staticmethod
    def _record_emergence(child_id: str, a_id: str, b_id: str, tick: int) -> None:
        payload = NodeEmergencePayload(
            node_id=child_id,
            origin_type="SIP_RECOMB",
            parents=[a_id, b_id],
        )
        entry = NodeEmergenceLog(tick=tick, value=payload)
        log_json("event", "node_emergence_log", entry.model_dump(), tick=tick)

    @staticmethod
    def _register_pending(child_id: str, a_id: str, b_id: str, tick: int) -> None:
        _sip_pending.append((child_id, [a_id, b_id], tick, "SIP_RECOMB"))
        global _sip_success_count
        _sip_success_count += 1
        for pid in (a_id, b_id):
            _spawn_counts[pid] = _spawn_counts.get(pid, 0) + 1

    def spawn_child(self, parent_a: Node, parent_b: Node, tick: int) -> None:
        """Spawn a recombination child node from ``parent_a`` and ``parent_b``."""
        if not Config.propagation_control.get("enable_sip_recomb", True):
            return
        if self._limit_reached(parent_a.id, parent_b.id):
            self._log_limit(parent_a, parent_b, tick)
            return
        child_id = f"{parent_a.id}_{parent_b.id}_R{tick}"
        if child_id in self.graph.nodes:
            return
        freq = (parent_a.frequency + parent_b.frequency) / 2 + np.random.normal(
            0, Config.SIP_MUTATION_SCALE
        )
        x = (parent_a.x + parent_b.x) / 2 + np.random.uniform(-5, 5)
        y = (parent_a.y + parent_b.y) / 2 + np.random.uniform(-5, 5)
        self.graph.add_node(
            child_id,
            x=x,
            y=y,
            frequency=freq,
            origin_type="SIP_RECOMB",
            generation_tick=tick,
            parent_ids=[parent_a.id, parent_b.id],
        )
        self.graph.add_edge(parent_a.id, child_id)
        self.graph.add_edge(parent_b.id, child_id)
        self._record_emergence(child_id, parent_a.id, parent_b.id, tick)
        self._register_pending(child_id, parent_a.id, parent_b.id, tick)


SIP_COHERENCE_DURATION = 3
SIP_DECOHERENCE_THRESHOLD = 0.5


def _spawn_sip_child(parent, tick: int) -> None:
    if not Config.propagation_control.get("enable_sip_child", True):
        return
    if _spawn_counts.get(parent.id, 0) >= Config.max_children_per_node > 0:
        if Config.is_log_enabled("event", "propagation_failure_log"):
            log_json(
                "event",
                "propagation_failure_log",
                {
                    "type": "SPAWN_LIMIT",
                    "parent": parent.id,
                    "origin_type": "SIP_BUD",
                },
                tick=tick,
            )
        return
    child_id = f"{parent.id}_S{tick}"
    if child_id in graph.nodes:
        return
    graph.add_node(
        child_id,
        x=parent.x + 10,
        y=parent.y + 10,
        frequency=parent.frequency,
        origin_type="SIP_BUD",
        generation_tick=tick,
        parent_ids=[parent.id],
    )
    graph.add_edge(parent.id, child_id)
    payload = NodeEmergencePayload(
        node_id=child_id,
        origin_type="SIP_BUD",
        parents=[parent.id],
    )
    entry = NodeEmergenceLog(tick=tick, value=payload)
    log_json("event", "node_emergence_log", entry.model_dump(), tick=tick)
    _update_growth_log(tick)
    _sip_pending.append((child_id, [parent.id], tick, "SIP_BUD"))
    global _sip_success_count
    _sip_success_count += 1
    _spawn_counts[parent.id] = _spawn_counts.get(parent.id, 0) + 1


def _spawn_sip_recomb_child(parent_a, parent_b, tick: int) -> None:
    SIPRecombinationService(graph).spawn_child(parent_a, parent_b, tick)


def _check_sip_failures(tick: int) -> None:
    global _sip_failure_count
    for child_id, parents, start, otype in list(_sip_pending):
        if tick - start < Config.SIP_STABILIZATION_WINDOW:
            continue
        node = graph.get_node(child_id)
        success = node and len(node.tick_history) > 0 and node.coherence > 0.5
        if not success:
            if node:
                graph.remove_node(child_id)
            for pid in parents:
                p = graph.get_node(pid)
                if p:
                    p.decoherence_debt += Config.SIP_FAILURE_ENTROPY_INJECTION
            log_json(
                "event",
                "propagation_failure_log",
                {
                    "type": "SIP_FAILURE",
                    "parent": parents[0],
                    "child": child_id,
                    "reason": (
                        "Insufficient coherence after "
                        f"{Config.SIP_STABILIZATION_WINDOW} ticks"
                    ),
                    "entropy_injected": Config.SIP_FAILURE_ENTROPY_INJECTION,
                    "origin_type": otype,
                },
                tick=tick,
            )
            _sip_failure_count += 1
        _sip_pending.remove((child_id, parents, start, otype))


def trigger_csp(parent_id: str, x: float, y: float, tick: int) -> None:
    if not Config.propagation_control.get("enable_csp", True):
        return
    for i in range(Config.CSP_MAX_NODES):
        seed_id = f"{parent_id}_CSPseed{i}_{tick}"
        dx = np.random.uniform(-Config.CSP_RADIUS, Config.CSP_RADIUS)
        dy = np.random.uniform(-Config.CSP_RADIUS, Config.CSP_RADIUS)
        _csp_seeds.append(
            {
                "id": seed_id,
                "parent": parent_id,
                "x": x + dx,
                "y": y + dy,
                "tick": tick,
            }
        )


def _process_csp_seeds(tick: int) -> None:
    global _csp_seeds, _csp_success_count, _csp_failure_count
    if not Config.propagation_control.get("enable_csp", True):
        _csp_seeds.clear()
        return
    service = CSPSeedService(graph)
    _csp_seeds = service.process(_csp_seeds, tick)
    _csp_success_count += service.success
    _csp_failure_count += service.failure


def _update_growth_log(tick: int) -> None:
    global _sip_success_count, _sip_failure_count, _csp_success_count, _csp_failure_count
    avg_coh = (
        sum(n.coherence for n in graph.nodes.values()) / len(graph.nodes)
        if graph.nodes
        else 0.0
    )
    payload = StructuralGrowthPayload(
        node_count=len(graph.nodes),
        edge_count=len(graph.edges) + len(graph.bridges),
        sip_success_total=_sip_success_count,
        csp_success_total=_csp_success_count,
        avg_coherence=round(avg_coh, 4),
    )
    entry = StructuralGrowthLog(tick=tick, value=payload)
    log_json("tick", "structural_growth_log", payload.model_dump(), tick=tick)
    _sip_success_count = 0
    _sip_failure_count = 0
    _csp_success_count = 0
    _csp_failure_count = 0


def check_propagation(tick: int) -> None:
    global _spawn_counts, _spawn_tick
    if _spawn_tick != tick:
        _spawn_counts = {}
        _spawn_tick = tick
    _check_sip_failures(tick)
    _process_csp_seeds(tick)
    if Config.propagation_control.get("enable_sip_child", True):
        for node in list(graph.nodes.values()):
            if (
                node.sip_streak >= SIP_COHERENCE_DURATION
                and node.decoherence_debt < SIP_DECOHERENCE_THRESHOLD
            ):
                _spawn_sip_child(node, tick)
                node.sip_streak = 0

    if Config.propagation_control.get("enable_sip_recomb", True):
        for bridge in graph.bridges:
            if not bridge.active or bridge.state.name != "STABLE":
                continue
            if bridge.trust_score < Config.SIP_RECOMB_MIN_TRUST:
                continue
            a = graph.get_node(bridge.node_a_id)
            b = graph.get_node(bridge.node_b_id)
            if not a or not b:
                continue
            if (
                a.sip_streak >= SIP_COHERENCE_DURATION
                and b.sip_streak >= SIP_COHERENCE_DURATION
                and a.decoherence_debt < SIP_DECOHERENCE_THRESHOLD
                and b.decoherence_debt < SIP_DECOHERENCE_THRESHOLD
            ):
                _spawn_sip_recomb_child(a, b, tick)
                a.sip_streak = 0
                b.sip_streak = 0
