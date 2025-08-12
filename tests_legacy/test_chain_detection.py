import numpy as np

from Causal_Web.config import Config
from Causal_Web.engine.models.graph import CausalGraph
from Causal_Web.engine.models.tick import Tick
from Causal_Web.engine.services.node_services import EdgePropagationService


def test_collect_chain_handles_cycles():
    g = CausalGraph()
    g.add_node("a")
    g.add_node("b")
    g.get_node("a").psi[:] = 0
    g.get_node("b").psi[:] = 0
    g.add_edge("a", "b")
    g.add_edge("b", "a")
    tick = Tick("self", 0.0, 1.0, 0.0)
    service = EdgePropagationService(
        node=g.get_node("a"),
        tick_time=0,
        phase=0.0,
        origin="self",
        graph=g,
        tick=tick,
    )
    edge = g.get_edges_from("a")[0]
    chain = service._collect_chain(edge)
    assert len(chain) == 1


def test_hadamard_chain_integration(monkeypatch):
    g = CausalGraph()
    for i in range(101):
        g.add_node(str(i))
        if i > 0:
            g.get_node(str(i)).psi[:] = 0
    for i in range(100):
        g.add_edge(str(i), str(i + 1), u_id=1)

    monkeypatch.setattr(Config, "chi_max", 2)

    tick = Tick("self", 0.0, 1.0, 0.0)
    service = EdgePropagationService(
        node=g.get_node("0"),
        tick_time=0,
        phase=0.0,
        origin="self",
        graph=g,
        tick=tick,
    )

    called = []
    original = EdgePropagationService._propagate_chain_mps

    def wrapped(self, chain, kappa):
        called.append(True)
        return original(self, chain, kappa)

    monkeypatch.setattr(EdgePropagationService, "_propagate_chain_mps", wrapped)

    service.propagate()
    assert called

    final = g.get_node("100").psi
    exact_amp = 2 ** (-50)
    err = abs(final[0] - exact_amp) / exact_amp
    assert err < 0.01
