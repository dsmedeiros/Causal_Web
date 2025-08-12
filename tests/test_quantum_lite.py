import numpy as np
import random

import pytest

from Causal_Web.engine.models.graph import CausalGraph
from Causal_Web.engine.services.node_services import EdgePropagationService
from Causal_Web.engine.models.tick import GLOBAL_TICK_POOL
from Causal_Web.engine.tick_engine.tick_router import TickRouter
from Causal_Web.config import Config

pytestmark = pytest.mark.skip(reason="legacy engine removed")


def _build_graph():
    g = CausalGraph()
    for nid in ["S", "A", "B", "D1", "D2"]:
        g.add_node(nid)
    g.add_edge("S", "A", attenuation=0.5)
    g.add_edge("S", "B", attenuation=0.5)
    g.add_edge("A", "D1")
    g.add_edge("A", "D2")
    g.add_edge("B", "D1")
    g.add_edge("B", "D2", phase_shift=np.pi)
    return g


def _run_sim(decohere: bool, runs: int = 200) -> float:
    Config.N_DECOH = 3
    rng = random.Random(0)
    np.random.seed(0)
    g = _build_graph()
    counts = np.zeros(2)
    for _ in range(runs):
        for node_id, node in g.nodes.items():
            node.psi = (
                np.array([1 + 0j, 0 + 0j])
                if node_id == "S"
                else np.zeros(2, dtype=np.complex128)
            )
            node.incoming_tick_counts.clear()
        tick = GLOBAL_TICK_POOL.acquire()
        tick.origin = "self"
        tick.time = 0
        tick.phase = 0
        tick.amplitude = 1
        EdgePropagationService(g.get_node("S"), 0, 0, "self", g, tick).propagate()
        GLOBAL_TICK_POOL.release(tick)
        if decohere:
            for sid in ["A", "B"]:
                node = g.get_node(sid)
                node.incoming_tick_counts[0] = Config.N_DECOH
                TickRouter.record_fanin(node, 0)
        for sid in ["A", "B"]:
            phase = rng.uniform(0, 2 * np.pi) if decohere else 0.0
            tick2 = GLOBAL_TICK_POOL.acquire()
            tick2.origin = sid
            tick2.time = 0
            tick2.phase = phase
            tick2.amplitude = 1
            EdgePropagationService(g.get_node(sid), 0, phase, sid, g, tick2).propagate()
            GLOBAL_TICK_POOL.release(tick2)
        counts[0] += abs(g.get_node("D1").psi[0]) ** 2
        counts[1] += abs(g.get_node("D2").psi[0]) ** 2
        for nid in ["A", "B", "D1", "D2"]:
            g.get_node(nid).psi[:] = 0
    probs = counts / runs
    return float(np.var(probs))


def test_double_slit_interference():
    var = _run_sim(decohere=False)
    assert var > 0.2


def test_double_slit_decoherence():
    var = _run_sim(decohere=True)
    assert var < 0.2
