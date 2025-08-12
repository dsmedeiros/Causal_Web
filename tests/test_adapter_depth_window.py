from Causal_Web.engine.engine_v2.adapter import EngineAdapter
from Causal_Web.engine.engine_v2.state import Packet


def test_run_until_next_window_or_rolls_window():
    graph = {
        "params": {"W0": 2},
        "nodes": [{"id": "0", "rho_mean": 0.0}],
        "edges": [{"from": "0", "to": "0", "delay": 1.0}],
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

    graph = {"nodes": [{"id": "0", "rho_mean": 0.0}], "edges": []}

    adapter = EngineAdapter()
    adapter.build_graph(graph)

    frame = adapter.run_until_next_window_or(limit=10)

    assert frame.depth == 0
    assert frame.events == 0


def test_run_until_next_window_or_single_event():
    """Processing a single event does not advance the depth."""

    graph = {"nodes": [{"id": "0", "rho_mean": 0.0}], "edges": []}

    adapter = EngineAdapter()
    adapter.build_graph(graph)

    adapter._scheduler.push(0, 0, 0, Packet(0, 0))
    frame = adapter.run_until_next_window_or(limit=10)

    assert frame.events == 1
    assert frame.depth == 0
    lccm = adapter._vertices[0]["lccm"]
    assert lccm.window_idx == 0


def test_run_until_next_window_or_stops_at_boundary():
    graph = {
        "params": {"W0": 2},
        "nodes": [{"id": "0", "rho_mean": 0.0}],
        "edges": [{"from": "0", "to": "0", "delay": 1.0}],
    }

    adapter = EngineAdapter()
    adapter.build_graph(graph)
    adapter._scheduler.push(0, 0, 0, Packet(0, 0))
    adapter._scheduler.push(0, 0, 0, Packet(0, 0))
    frame = adapter.run_until_next_window_or(limit=10)

    assert frame.window == 1
    assert len(adapter._scheduler) > 0


def test_run_until_next_window_or_any_vertex_boundary():
    """Stop once any vertex crosses a window boundary.

    `run_until_next_window_or` should halt as soon as *one* vertex's
    `window_idx` rolls forward.  Other vertices may still have pending events,
    which must remain queued for the next call.  This behaviour preserves
    per-window isolation and prevents overshooting across unrelated vertices.
    """

    graph = {
        "params": {"W0": 2},
        "nodes": [
            {"id": "0", "rho_mean": 0.0},
            {"id": "1", "rho_mean": 0.0},
        ],
        "edges": [{"from": "0", "to": "0", "delay": 1.0}],
    }

    adapter = EngineAdapter()
    adapter.build_graph(graph)

    adapter._scheduler.push(0, 0, 0, Packet(0, 0))
    adapter._scheduler.push(3, 1, 0, Packet(1, 1))

    frame = adapter.run_until_next_window_or(limit=10)

    lccm0 = adapter._vertices[0]["lccm"]
    lccm1 = adapter._vertices[1]["lccm"]
    assert lccm0.window_idx == 1
    assert lccm1.window_idx == 0
    remaining = [
        key[0] for bucket in adapter._scheduler._buckets.values() for key, _ in bucket
    ]
    assert 1 in remaining
    assert frame.window == 1
