from Causal_Web.engine.engine_v2.adapter import EngineAdapter
from Causal_Web.engine.engine_v2.state import Packet
from Causal_Web.config import Config


def build_graph():
    return {
        "nodes": [
            {"id": "A", "window_len": 1},
            {"id": "B", "window_len": 1},
        ],
        "edges": [
            {"from": "A", "to": "B", "delay": 1.0, "density": 0.0},
        ],
        "params": {"W0": 1},
    }


def test_residual_updates_on_window_close():
    Config.rho_delay = {
        "alpha_d": 0.0,
        "alpha_leak": 0.0,
        "eta": 0.0,
        "gamma": 0.0,
        "rho0": 1.0,
    }
    adapter = EngineAdapter()
    adapter.build_graph(build_graph())
    adapter._scheduler.push(0, 0, 0, Packet(src=-1, dst=0, payload=None))
    adapter.run_until_next_window_or(None)
    adapter.run_until_next_window_or(None)
    snap = adapter.snapshot_for_ui()
    assert snap.counters["residual"] > 0.0
    assert snap.invariants["inv_conservation_residual"] == snap.counters["residual"]
