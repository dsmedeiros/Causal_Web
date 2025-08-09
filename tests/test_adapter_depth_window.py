from Causal_Web.engine.engine_v2.adapter import EngineAdapter
from Causal_Web.engine.engine_v2.state import Packet


def test_run_until_next_window_or_rolls_window():
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
    frame = adapter.run_until_next_window_or(limit=10)

    # Window index should have rolled once and depth advanced to W0
    lccm = adapter._vertices[0]["lccm"]
    assert lccm.window_idx == 1
    assert frame.depth == 2
    assert frame.events == 3


def test_run_until_next_window_or_no_events():
    """Adapter reports zero depth and events when scheduler is empty."""

    graph = {"vertices": [{"id": 0, "rho_mean": 0.0, "edges": []}]}

    adapter = EngineAdapter()
    adapter.build_graph(graph)

    frame = adapter.run_until_next_window_or(limit=10)

    assert frame.depth == 0
    assert frame.events == 0


def test_run_until_next_window_or_single_event():
    """Processing a single event does not advance the depth."""

    graph = {"vertices": [{"id": 0, "rho_mean": 0.0, "edges": []}]}

    adapter = EngineAdapter()
    adapter.build_graph(graph)

    adapter._scheduler.push(0, 0, 0, Packet(0, 0))
    frame = adapter.run_until_next_window_or(limit=10)

    assert frame.events == 1
    assert frame.depth == 0
    lccm = adapter._vertices[0]["lccm"]
    assert lccm.window_idx == 0
