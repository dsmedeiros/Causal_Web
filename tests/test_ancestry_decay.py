import numpy as np

from Causal_Web.engine.engine_v2.adapter import EngineAdapter
from Causal_Web.engine.engine_v2.state import Packet


def test_delta_m_decay_no_q_arrivals():
    adapter = EngineAdapter()
    graph = {
        "params": {"W0": 2},
        "nodes": [{"id": "0"}],
        "edges": [{"from": "0", "to": "0", "delay": 1.0}],
    }
    adapter.build_graph(graph)

    v_arr = adapter._arrays.vertices
    v_arr["m0"][0] = 1.0
    v_arr["m1"][0] = 0.0
    v_arr["m2"][0] = 0.0
    v_arr["m_norm"][0] = 1.0

    lccm = adapter._vertices[0]["lccm"]
    lccm.layer = "Θ"

    adapter._scheduler.push(0, 0, 0, Packet(src=0, dst=0))
    adapter.run_until_next_window_or(limit=10)

    m = np.array([v_arr["m0"][0], v_arr["m1"][0], v_arr["m2"][0]])
    assert np.allclose(m, np.array([1.0, 0.0, 0.0]))
    assert v_arr["m_norm"][0] < 1.0

    adapter._update_ancestry(0, 0, 0, 0, np.pi / 2, 1.0)

    m = np.array([v_arr["m0"][0], v_arr["m1"][0], v_arr["m2"][0]])
    assert m[1] > 0.0


def test_lambda_q_downweights_only_q_arrivals():
    adapter = EngineAdapter()
    adapter.build_graph({"nodes": [{"id": "0"}], "edges": []})

    v_arr = adapter._arrays.vertices
    v_arr["m0"][0] = 1.0
    v_arr["m1"][0] = 0.0
    v_arr["m2"][0] = 0.0
    v_arr["m_norm"][0] = 1.0

    lccm = adapter._vertices[0]["lccm"]
    lccm._lambda = 100
    lccm._lambda_q = 0

    adapter._update_ancestry(0, 0, 0, 0, np.pi / 2, 1.0)

    m = np.array([v_arr["m0"][0], v_arr["m1"][0], v_arr["m2"][0]])
    assert m[1] > 0.05
