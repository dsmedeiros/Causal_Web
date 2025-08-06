import math

from Causal_Web.engine.models.graph import CausalGraph


def test_cnot_source_creates_epsilon_edges():
    g = CausalGraph()
    g.add_node("J", cnot_source=True)
    g.add_node("A")
    g.add_node("B")
    g.add_edge("J", "A")
    g.add_edge("J", "B")
    j = g.get_node("J")
    j.apply_tick(0, 0.0, g)
    edges = g.get_edges_from("J")
    assert edges[0].epsilon and edges[1].epsilon
    assert edges[0].partner is edges[1]


def test_gauge_phase_shift():
    g = CausalGraph()
    g.add_node("A")
    g.add_node("B")
    g.add_edge("A", "B", A_phase=math.pi / 2)
    a = g.get_node("A")
    a.apply_tick(0, 0.0, g)
    b = g.get_node("B")
    incoming = b.incoming_phase_queue[min(b.incoming_phase_queue.keys())][0][0]
    assert math.isclose(incoming, math.pi / 2, rel_tol=1e-5)
