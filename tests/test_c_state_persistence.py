import numpy as np

from Causal_Web.engine.engine_v2.adapter import EngineAdapter
from Causal_Web.engine.engine_v2.state import Packet


def test_c_state_persists_across_windows():
    adapter = EngineAdapter()
    graph = {"nodes": [{"id": "0", "window_len": 1}], "edges": []}
    adapter.build_graph(graph)
    v_arr = adapter._arrays.vertices
    lccm = adapter._vertices[0]["lccm"]
    lccm.layer = "C"
    v_arr["bit"][0] = 1
    v_arr["conf"][0] = 0.5
    adapter._vertices[0]["bit_deque"].extend([1, 0, 1])
    window = lccm._window_size()
    adapter._scheduler.push(window, 0, -1, Packet(src=0, dst=0, payload={"bit": 1}))
    adapter.run_until_next_window_or(None)
    assert v_arr["bit"][0] == 1
    assert np.isclose(v_arr["conf"][0], 0.5)
    assert not adapter._vertices[0]["bit_deque"]
