from __future__ import annotations

from ..models.node import Node
from ..models.tick import Tick
import numpy as np
from ...config import Config


class TickRouter:
    """Route ticks across LCCM layers"""

    LAYERS = [
        "tick",
        "phase",
        "collapse",
        "bridge",
        "coherence",
        "decoherence",
        "law-wave",
    ]

    @classmethod
    def next_layer(cls, current: str) -> str:
        try:
            idx = cls.LAYERS.index(current)
            if idx < len(cls.LAYERS) - 1:
                return cls.LAYERS[idx + 1]
        except ValueError:
            pass
        return cls.LAYERS[-1]

    @classmethod
    def record_fanin(cls, node: Node, tick_time: int, graph=None) -> None:
        """Increment fan-in counts and collapse via Born rule.

        ``graph`` is optionally supplied so that entanglement constraints can
        propagate collapses across ``\u03b5`` edges.
        """
        count = node.incoming_tick_counts[tick_time]
        if count == getattr(Config, "N_DECOH", 0):
            probs = np.abs(node.psi) ** 2
            if probs.sum() == 0:
                return
            outcome = np.random.choice(2, p=probs / probs.sum())
            if outcome == 0:
                node.psi = np.array([1 + 0j, 0 + 0j], np.complex128)
            else:
                node.psi = np.array([0 + 0j, 1 + 0j], np.complex128)
            node.collapse_origin[tick_time] = node.collapse_origin.get(
                tick_time, "self"
            )
            if graph is not None:
                from ..services.entanglement_service import EntanglementService

                EntanglementService.collapse_epsilon(graph, node, tick_time)
            node.incoming_tick_counts[tick_time] = 0

    @classmethod
    def route_tick(cls, node: Node, tick: Tick) -> None:
        """Update ``tick`` to the next layer and record the transition."""
        from ..logging.logger import log_json

        new_layer = cls.next_layer(tick.layer)
        if new_layer != tick.layer:
            record = {
                "tick": tick.time,
                "node": node.id,
                "from": tick.layer,
                "to": new_layer,
                "trace_id": tick.trace_id,
            }
            log_json(
                "event",
                "layer_transition_log",
                {
                    "node": node.id,
                    "from": tick.layer,
                    "to": new_layer,
                    "trace_id": tick.trace_id,
                },
                tick=tick.time,
            )
            tick.layer = new_layer
