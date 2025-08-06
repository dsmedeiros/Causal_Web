from __future__ import annotations

import numpy as np

from ..models.node import Node


class EntanglementService:
    """Utilities for managing entanglement effects."""

    @staticmethod
    def collapse_epsilon(graph, node: Node, tick_time: int) -> None:
        """Collapse epsilon-linked partners to the opposite eigenstate.

        Both incoming and outgoing ``\u03b5`` edges are inspected so collapse
        propagates even when only a single directed edge exists.

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

        visited: set[str] = set()
        edges = graph.get_edges_from(node.id) + graph.get_edges_to(node.id)
        for edge in edges:
            if not getattr(edge, "epsilon", False):
                continue
            partner_id = edge.target if edge.source == node.id else edge.source
            if partner_id in visited:
                continue
            partner = graph.get_node(partner_id)
            if partner is None or partner.collapse_origin.get(tick_time) is not None:
                continue
            partner.psi = np.array([node.psi[1], node.psi[0]], np.complex128)
            partner.collapse_origin[tick_time] = "epsilon"
            partner.incoming_tick_counts[tick_time] = 0
            visited.add(partner_id)
