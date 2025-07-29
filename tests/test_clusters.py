import math
from Causal_Web.engine.models.graph import CausalGraph
from Causal_Web.engine.models.node import Node


def test_hierarchical_cluster_assignment():
    g = CausalGraph()
    g.add_node("A")
    g.add_node("B")
    g.nodes["A"].law_wave_frequency = 1.0
    g.nodes["B"].law_wave_frequency = 1.02
    g.nodes["A"].coherence = 0.95
    g.nodes["B"].coherence = 0.95
    clusters = g.hierarchical_clusters()
    a_cluster = g.nodes["A"].cluster_ids.get(0)
    b_cluster = g.nodes["B"].cluster_ids.get(0)
    assert a_cluster == b_cluster
    assert 0 in g.nodes["A"].cluster_ids
    assert 1 in clusters
