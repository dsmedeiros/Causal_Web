import pytest

from Causal_Web.engine.engine_v2.adapter import EngineAdapter
from Causal_Web.engine.engine_v2.state import Packet
from Causal_Web.config import Config


def build_graph():
    return {
        "nodes": [
            {"id": "A", "window_len": 1},
            {"id": "B", "window_len": 1},
            {"id": "C", "window_len": 1},
        ],
        "edges": [
            {"from": "A", "to": "B", "delay": 1.0, "density": 0.0},
            {"from": "B", "to": "C", "delay": 1.0, "density": 0.0},
        ],
        "params": {"W0": 1},
    }


def test_incoming_injection_only_heats_delivering_edge():
    Config.rho_delay = {
        "alpha_d": 0.0,
        "alpha_leak": 0.0,
        "eta": 1.0,
        "gamma": 0.0,
        "rho0": 1.0,
        "inject_mode": "incoming",
    }
    adapter = EngineAdapter()
    adapter.build_graph(build_graph())
    adapter._scheduler.push(0, 0, -1, Packet(src=-1, dst=0, payload=None))
    adapter.run_until_next_window_or(None)
    arrays = adapter._arrays
    assert arrays is not None
    assert arrays.edges["rho"][0] == pytest.approx(1.0)
    assert arrays.edges["rho"][1] == pytest.approx(0.0)
