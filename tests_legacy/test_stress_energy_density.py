import math

from Causal_Web.engine.fields.density import get_field
from legacy.engine.models.graph import CausalGraph


def test_density_increases_delay():
    field = get_field()
    field.clear()
    g = CausalGraph()
    for n in ["A", "B"]:
        g.add_node(n)
    g.add_edge("A", "B", delay=1)
    edge = g.get_edges_from("A")[0]
    field.deposit(edge, 1.0)
    kappa = 1.0
    base = edge.adjusted_delay(1.0, 1.0, kappa, graph=g)
    rho = field.get(edge)
    delay = base * (1 + kappa * rho)
    assert delay == 2


def test_diffusion_spreads_density():
    field = get_field()
    field.clear()
    g = CausalGraph()
    for n in ["A", "B", "C"]:
        g.add_node(n)
    g.add_edge("A", "B")
    g.add_edge("B", "C")
    e1 = g.get_edges_from("A")[0]
    e2 = g.get_edges_from("B")[0]
    field.deposit(e1, 1.0)
    field.diffuse(g, alpha=0.5)
    assert field.get(e2) > 0.0
