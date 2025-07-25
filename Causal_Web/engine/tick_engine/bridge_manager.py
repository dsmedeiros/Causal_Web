"""Dynamic bridge formation utilities."""

from __future__ import annotations

import math

from ...config import Config
from ..graph import CausalGraph

_graph: CausalGraph | None = None


def attach_graph(graph: CausalGraph) -> None:
    global _graph
    _graph = graph


def _bridge_thresholds(global_tick: int) -> tuple[float, float]:
    if global_tick < 20:
        return 0.6, 0.5
    if global_tick < 50:
        progress = (global_tick - 20) / 30
        coh = 0.6 + progress * (0.9 - 0.6)
        drift = 0.5 - progress * (0.5 - 0.1)
        return coh, drift
    return 0.9, 0.1


def dynamic_bridge_management(global_tick: int) -> None:
    assert _graph is not None
    existing = {(b.node_a_id, b.node_b_id) for b in _graph.bridges}
    existing |= {(b.node_b_id, b.node_a_id) for b in _graph.bridges}
    coherence_thresh, drift_thresh = _bridge_thresholds(global_tick)
    for a in _graph.nodes.values():
        for b in _graph.nearby_nodes(a):
            if a.id >= b.id:
                continue
            if (a.id, b.id) in existing:
                continue
            if a.cluster_ids.get(0) != b.cluster_ids.get(0):
                continue
            drift = abs((a.phase - b.phase + math.pi) % (2 * math.pi) - math.pi)
            if (
                drift < drift_thresh
                and a.coherence > coherence_thresh
                and b.coherence > coherence_thresh
            ):
                _graph.add_bridge(a.id, b.id, formed_at_tick=global_tick, seeded=False)
                bridge = _graph.bridges[-1]
                bridge._log_dynamics(
                    global_tick,
                    "formed",
                    {
                        "phase_delta": drift,
                        "coherence": (a.coherence + b.coherence) / 2,
                    },
                )
                bridge.update_state(global_tick)
