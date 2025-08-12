from Causal_Web.engine.engine_v2.adapter import EngineAdapter
from Causal_Web.engine.engine_v2.state import Packet


def test_adapter_step_processes_packet() -> None:
    graph = {
        "params": {"W0": 1},
        "nodes": [{"id": "0"}, {"id": "1"}],
        "edges": [{"from": "0", "to": "1", "delay": 1.0}],
    }
    adapter = EngineAdapter()
    adapter.build_graph(graph)
    adapter._scheduler.push(0, 0, 0, Packet(0, 1))
    frame = adapter.step(max_events=1)
    assert frame.events == 1
