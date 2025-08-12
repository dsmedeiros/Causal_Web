import numpy as np

from Causal_Web.engine.engine_v2.adapter import EngineAdapter
from Causal_Web.engine.engine_v2.state import Packet
from Causal_Web.config import Config


def build_simple_graph():
    graph = {
        "nodes": [
            {"id": "A", "window_len": 1},
            {"id": "B", "window_len": 1},
        ],
        "edges": [
            {"from": "A", "to": "B", "delay": 1.0, "density": 0.0},
        ],
        "params": {"W0": 1},
    }
    return graph


def test_rho_to_d_eff_and_eq():
    Config.rho_delay = {
        "alpha_d": 0.0,
        "alpha_leak": 0.0,
        "eta": 0.0,
        "gamma": 0.0,
        "rho0": 1.0,
    }
    adapter = EngineAdapter()
    adapter.build_graph(build_simple_graph())
    # seed an initial packet to vertex A
    adapter._scheduler.push(0, 0, 0, Packet(src=-1, dst=0, payload=None))
    adapter.run_until_next_window_or(None)
    # d_eff updated on edge
    arrays = adapter._arrays
    assert arrays is not None
    assert int(arrays.edges["d_eff"][0]) == 1

    # process packet delivered to B, triggering window close
    adapter.run_until_next_window_or(None)
    eq_val = arrays.vertices["EQ"][1]
    assert eq_val == 1.0
    assert np.allclose(arrays.vertices["psi_acc"][1], 0.0)
    assert adapter._vertices[1]["lccm"]._eq == 1.0
