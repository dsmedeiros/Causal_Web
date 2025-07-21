import json
from Causal_Web.graph.io import load_graph, save_graph, new_graph


def test_load_and_save_roundtrip(tmp_path):
    data = {
        "nodes": {"A": {"x": 0, "y": 0}},
        "edges": [{"from": "A", "to": "A"}],
    }
    path = tmp_path / "g.json"
    path.write_text(json.dumps(data))

    graph = load_graph(str(path))
    assert "A" in graph.nodes

    out = tmp_path / "out.json"
    save_graph(str(out), graph)
    saved = json.loads(out.read_text())
    assert saved["nodes"]["A"]["x"] == 0


def test_new_graph_default_node():
    graph = new_graph(True)
    assert graph.nodes

