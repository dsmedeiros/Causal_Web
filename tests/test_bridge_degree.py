from Causal_Web.engine.engine_v2.adapter import EngineAdapter
from Causal_Web.engine.engine_v2.state import Packet


def test_bridges_contribute_to_degree():
    adapter = EngineAdapter()
    graph = {"nodes": [{"id": "0"}, {"id": "1"}], "edges": []}
    adapter.build_graph(graph)
    adapter._epairs._create_bridge(0, 1, d_bridge=1)
    lccm = adapter._vertices[0]["lccm"]
    window = lccm._window_size()
    adapter._scheduler.push(window, 0, -1, Packet(src=0, dst=0))
    adapter.run_until_next_window_or(None)
    assert adapter._vertices[0]["lccm"].deg == adapter._vertices[0]["base_deg"] + 1


def test_bridges_adjust_window_size():
    adapter = EngineAdapter()
    graph = {
        "params": {"W0": 2, "zeta1": 2.0},
        "nodes": [{"id": "0"}, {"id": "1"}],
        "edges": [],
    }
    adapter.build_graph(graph)
    lccm = adapter._vertices[0]["lccm"]
    base_deg = adapter._vertices[0]["base_deg"]
    w0 = lccm._window_size()

    adapter._epairs._create_bridge(0, 1, d_bridge=1)
    lccm.deg = base_deg + len(adapter._epairs.partners(0))
    w1 = lccm._window_size()
    assert w1 > w0

    adapter._epairs._remove_bridge(0, 1)
    lccm.deg = base_deg + len(adapter._epairs.partners(0))
    w2 = lccm._window_size()
    assert w2 == w0
