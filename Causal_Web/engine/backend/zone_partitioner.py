from __future__ import annotations

"""Greedy partitioning of classical nodes into zones."""

from typing import Any, Iterable, List, Set

import networkx as nx


def partition_zones(graph: Any) -> List[Set[str]]:
    """Return connected components of classical nodes in ``graph``.

    Nodes marked with ``is_classical`` are treated as classical and grouped
    into zones separated by coherent quantum nodes. The function returns a
    list of node identifier sets, one for each classical zone.
    """

    g = nx.Graph()

    # Handle both NetworkX graphs and CausalGraph objects.
    if hasattr(graph, "nodes") and isinstance(graph.nodes, dict):
        for nid, node in graph.nodes.items():
            if getattr(node, "is_classical", False):
                g.add_node(nid)
        for edge in getattr(graph, "edges", []):
            src = getattr(edge, "source", None)
            tgt = getattr(edge, "target", None)
            if g.has_node(src) and g.has_node(tgt):
                g.add_edge(src, tgt)
    else:  # assume NetworkX style
        for nid, data in graph.nodes(data=True):
            if data.get("is_classical"):
                g.add_node(nid)
        for src, tgt in graph.edges():
            if g.has_node(src) and g.has_node(tgt):
                g.add_edge(src, tgt)

    return [set(c) for c in nx.connected_components(g)]
