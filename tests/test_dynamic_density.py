import math
from Causal_Web.engine.graph import CausalGraph
from Causal_Web.engine.node import Edge


def test_compute_local_density():
    g = CausalGraph()
    for n in ["A", "B", "C"]:
        g.add_node(n)
    g.add_edge("A", "B")
    g.add_edge("B", "C")
    edge = g.get_edges_from("A")[0]
    density = g.compute_local_density(edge, radius=1)
    assert math.isclose(density, 2.0)


def test_dynamic_delay_uses_computed_density():
    g = CausalGraph()
    for n in ["A", "B", "C"]:
        g.add_node(n)
    g.add_edge("A", "B", delay=2, density=None)
    g.add_edge("B", "C")
    g.precompute_local_densities(radius=1)
    edge = g.get_edges_from("A")[0]
    delay = edge.adjusted_delay(1.0, 1.0, kappa=1.0, graph=g)
    assert delay == int(2 * (1 + 2))
