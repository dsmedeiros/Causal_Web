from __future__ import annotations

import numpy as np

from ..models.node import Node


class EntanglementService:
    """Utilities for managing entanglement effects."""

    @staticmethod
    def collapse_epsilon(graph, node: Node, tick_time: int) -> None:
        """Collapse epsilon-linked partners to the opposite eigenstate.

        Parameters
        ----------
        graph:
            The :class:`~Causal_Web.engine.models.graph.CausalGraph` containing
            the nodes.
        node:
            Node that has just collapsed.
        tick_time:
            Global tick index of the collapse event.
        """

        for edge in graph.get_edges_from(node.id):
            if not getattr(edge, "epsilon", False):
                continue
            partner = graph.get_node(edge.target)
            if partner is None:
                continue
            if partner.collapse_origin.get(tick_time) is not None:
                continue
            partner.psi = np.array([node.psi[1], node.psi[0]], np.complex128)
            partner.collapse_origin[tick_time] = "epsilon"
            partner.incoming_tick_counts[tick_time] = 0
