import json
import math

import numpy as np

from Causal_Web.engine.models.graph import CausalGraph
from Causal_Web.engine.services import (
    EntanglementService,
    GraphLoadService,
)


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


def test_epsilon_collapse_copies_probabilities():
    g = CausalGraph()
    g.add_node("A")
    g.add_node("B")
    g.add_edge("A", "B", epsilon=True, partner_id="p")
    g.add_edge("B", "A", epsilon=True, partner_id="p")
    a = g.get_node("A")
    b = g.get_node("B")
    a.psi = np.array([1, 0], np.complex128)
    a.probabilities = np.array([0.2, 0.8])
    b.psi = np.array([0, 1], np.complex128)
    b.probabilities = np.array([0.5, 0.5])
    EntanglementService.collapse_epsilon(g, a, 0)
    assert np.allclose(b.probabilities, [0.8, 0.2])


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


def test_cnot_source_round_trip(tmp_path):
    g = CausalGraph()
    g.add_node("J", cnot_source=True)
    g.add_node("A")
    g.add_node("B")
    g.add_edge("J", "A")
    g.add_edge("J", "B")
    j = g.get_node("J")
    j.apply_tick(0, 0.0, g)
    data = g.to_dict()
    path = tmp_path / "graph.json"
    path.write_text(json.dumps(data))
    g2 = CausalGraph()
    GraphLoadService(g2, str(path)).load()
    j2 = g2.get_node("J")
    assert j2.cnot_source
    edges = g2.get_edges_from("J")
    assert edges[0].partner is edges[1]
