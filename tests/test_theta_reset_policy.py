import numpy as np

from Causal_Web.config import Config
from Causal_Web.engine.engine_v2.adapter import EngineAdapter
from Causal_Web.engine.engine_v2.state import Packet


def _run_policy(policy: str) -> np.ndarray:
    original_theta = Config.theta_reset
    original_windowing = Config.windowing.copy()
    try:
        Config.theta_reset = policy
        Config.windowing["W0"] = 2
        Config.windowing["Dp"] = 2
        graph = {
            "params": {"W0": 2},
            "nodes": [{"id": "0", "rho_mean": 0.0}],
            "edges": [{"from": "0", "to": "0", "delay": 1.0}],
        }
        adapter = EngineAdapter()
        adapter.build_graph(graph)
        payload = {"p": np.array([1.0, 0.0], dtype=np.float32)}
        adapter._scheduler.push(0, 0, 0, Packet(src=0, dst=0, payload=payload))
        adapter.run_until_next_window_or(limit=10)
        return adapter._vertices[0]["p_v"].copy()
    finally:
        Config.theta_reset = original_theta
        Config.windowing = original_windowing


def test_theta_reset_uniform():
    p_final = _run_policy("uniform")
    assert np.allclose(p_final, [0.5, 0.5])


def test_theta_reset_hold():
    p_final = _run_policy("hold")
    assert np.allclose(p_final, [0.5, 0.5])


def test_theta_reset_renorm():
    p_final = _run_policy("renorm")
    assert np.allclose(p_final, [0.5, 0.5])
