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
