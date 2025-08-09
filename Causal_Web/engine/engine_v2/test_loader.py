import pytest

from Causal_Web.engine.engine_v2.loader import load_graph_arrays
from Causal_Web.config import Config


@pytest.fixture
def patch_config(monkeypatch):
    monkeypatch.setattr(
        Config,
        "windowing",
        {"W0": 3.0, "Q": 5, "Dq": 2, "Dp": 3},
    )
    monkeypatch.setattr(
        Config,
        "unitaries",
        {"u1": [[0.0, 1.0], [1.0, 0.0]]},
        raising=False,
    )

    yield


def test_load_graph_arrays_struct_and_defaults(patch_config):
    graph_json = {
        "nodes": [
            {"id": "A", "window_len": 10.0, "layer": 2},
            {"id": "B"},
            {"id": "C", "extra": 123},
        ],
        "edges": [
            {
                "from": "A",
                "to": "B",
                "weight": 2,
                "delay": 1,
                "density": 0.3,
                "phase_shift": 0.1,
                "A_phase": 0.2,
                "u_id": "u1",
            },
            {"from": "B", "to": "A", "density": 0.5},
            {"from": "B", "to": "B", "weight": 3, "delay": 2},
            {"from": "C", "to": "D"},
        ],
    }

    arrays = load_graph_arrays(graph_json)

    assert arrays.id_map == {"A": 0, "B": 1, "C": 2}

    v = arrays.vertices
    assert v["window_len"] == [10.0, 3.0, 3.0]
    assert v["layer"] == [2, 5, 5]
    assert all(len(row) == 2 for row in v["psi"])
    assert v["p"][0] == [1 / 3, 1 / 3, 1 / 3]

    e = arrays.edges
    assert e["src"] == [0, 1, 1]
    assert e["dst"] == [1, 0, 1]
    assert e["d0"] == [1, 0.0, 2]
    assert e["alpha"] == [2, 1.0, 3]
    assert e["rho"] == [0.3, 0.5, 0.0]
    assert e["U"][0] == [[0.0, 1.0], [1.0, 0.0]]
    assert e["U"][1] == [[1.0, 0.0], [0.0, 1.0]]
    assert e["sigma"] == [0.0, 0.0, 0.0]

    adj = arrays.adjacency
    assert adj["nbr_ptr"] == [0, 2, 4, 6]
    assert adj["nbr_idx"] == [1, 2, 0, 2, 0, 1]


def test_load_graph_arrays_numeric_types(patch_config):
    graph_json = {
        "nodes": [{"id": "A"}],
        "edges": [{"from": "A", "to": "A"}],
    }
    arrays = load_graph_arrays(graph_json)

    v = arrays.vertices
    assert all(isinstance(x, int) for x in v["depth"])
    assert all(isinstance(x, float) for x in v["window_len"])
    assert all(isinstance(x, int) for x in v["layer"])
    assert all(isinstance(x, int) for x in v["bit"])
    assert all(isinstance(x, float) for x in v["EQ"])

    e = arrays.edges
    numeric_fields = ["d0", "rho", "alpha", "phi", "A", "sigma"]
    for field in numeric_fields:
        assert all(isinstance(x, (int, float)) for x in e[field])
