import math
from legacy.engine.models.graph import CausalGraph
from Causal_Web.config import Config


def test_tick_saturation_density():
    g = CausalGraph()
    g.add_node("A")
    g.add_node("B")
    g.add_edge("A", "B")
    edge = g.get_edges_from("A")[0]
    Config.density_calc = "local_tick_saturation"
    edge.propagate_phase(0.0, 0, g)
    rho = g.compute_edge_density(edge)
    assert rho > 0
