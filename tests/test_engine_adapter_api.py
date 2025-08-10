from Causal_Web.engine.engine_v2.adapter import EngineAdapter
from Causal_Web.engine.engine_v2.state import (
    EdgeArray,
    Packet,
    TelemetryFrame,
    VertexArray,
)


def test_step_returns_telemetry_frame():
    graph = {
        "params": {"W0": 2},
        "vertices": [
            {
                "id": 0,
                "rho_mean": 0.0,
                "edges": [{"id": 0, "dst": 0, "d_eff": 1}],
            }
        ],
    }
    adapter = EngineAdapter()
    adapter.build_graph(graph)
    adapter._scheduler.push(0, 0, 0, Packet(0, 0))
    frame = adapter.step(max_events=1)

    assert isinstance(frame, TelemetryFrame)
    assert frame.events == 1
    assert frame.packets and isinstance(frame.packets[0], Packet)
    assert adapter.snapshot_for_ui()["depth"] == frame.depth
    assert adapter.current_depth() == frame.depth
    assert isinstance(frame.window, int)
    assert adapter.current_frame() == 1


def test_state_data_structures_defaults():
    vertices = VertexArray()
    edges = EdgeArray()
    pkt = Packet(src=1, dst=2, payload={"x": 1})
    frame = TelemetryFrame(depth=3, events=1, packets=[pkt], window=2)

    assert vertices.ids == []
    assert edges.ids == []
    assert frame.packets[0].payload == {"x": 1}
    assert frame.window == 2
