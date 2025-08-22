from Causal_Web.engine.engine_v2.adapter import EngineAdapter
from Causal_Web.engine.engine_v2.state import Packet


def test_delta_includes_micro_counters() -> None:
    adapter = EngineAdapter()
    graph = {
        "params": {"W0": 1},
        "nodes": [{"id": "0"}, {"id": "1"}],
        "edges": [{"from": "0", "to": "1", "delay": 1.0}],
    }
    adapter.build_graph(graph)
    adapter._scheduler.push(0, 0, 0, Packet(0, 1))
    adapter.step(max_events=1)
    delta = adapter._build_delta()
    assert delta is not None
    counters = delta["counters"]
    for key in [
        "windows_closed",
        "bridges_active",
        "events_processed",
        "edges_traversed",
        "residual_ewma",
        "residual_max",
    ]:
        assert key in counters
    invariants = delta["invariants"]
    assert "inv_no_signaling_delta" in invariants
