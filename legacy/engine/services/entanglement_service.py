from __future__ import annotations

"""Minimal entanglement helpers for the v2 engine."""

from typing import Any

from ..models.node import Node


class EntanglementService:
    """Placeholder utilities for epsilon-link behaviour.

    The legacy engine supported ``\u03b5``-linked nodes that collapsed in
    tandem. The event-driven v2 kernel has no direct equivalent, so these
    helpers now perform no work but remain for API compatibility.
    """

    @staticmethod
    def collapse_epsilon(graph: Any, node: Node, tick_time: int | None = None) -> None:
        """No-op handler for collapsing epsilon-linked partners.

        Parameters
        ----------
        graph:
            Graph instance containing ``node``.
        node:
            Node that would have triggered a collapse.
        tick_time:
            Retained for backward compatibility; ignored.
        """

        return None
