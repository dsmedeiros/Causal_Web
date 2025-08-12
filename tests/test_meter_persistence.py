import numpy as np

from Causal_Web.engine.engine_v2.adapter import EngineAdapter
from Causal_Web.engine.engine_v2.state import Packet


def test_theta_c_meters_persist():
    adapter = EngineAdapter()
    graph = {"nodes": [{"id": "0"}], "edges": []}
    adapter.build_graph(graph)
    lccm = adapter._vertices[0]["lccm"]
    window = lccm._window_size()
    adapter._scheduler.push(window, 0, -1, Packet(src=0, dst=0))
    adapter.run_until_next_window_or(None)
    v_arr = adapter._arrays.vertices
    assert np.isclose(v_arr["E_theta"][0], lccm.k_theta)
    assert np.isclose(v_arr["E_C"][0], lccm.k_c)
    assert np.isclose(v_arr["E_rho"][0], 0.0)
