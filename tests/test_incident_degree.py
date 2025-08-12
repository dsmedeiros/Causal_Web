from math import floor, log

from Causal_Web.engine.engine_v2.adapter import EngineAdapter


def test_incident_degree_used_for_window_size():
    """Incident degree combines in- and out-edges for window sizing."""
    graph = {
        "params": {"W0": 2, "zeta1": 1.0, "zeta2": 0.0},
        "nodes": [
            {"id": "0", "rho_mean": 0.0},
            {"id": "1", "rho_mean": 0.0},
            {"id": "2", "rho_mean": 0.0},
        ],
        "edges": [
            {"from": "1", "to": "0", "delay": 1.0},
            {"from": "0", "to": "2", "delay": 1.0},
        ],
    }

    adapter = EngineAdapter()
    adapter.build_graph(graph)

    lccm = adapter._vertices[0]["lccm"]
    assert lccm.deg == 2

    expected_w = 2 + floor(log(1 + 2))
    assert lccm._window_size() == expected_w

    lccm.advance_depth(expected_w - 1)
    assert lccm.window_idx == 0
    lccm.advance_depth(expected_w)
    assert lccm.window_idx == 1
