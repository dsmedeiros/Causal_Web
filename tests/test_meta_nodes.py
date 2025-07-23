import json
from Causal_Web.graph.model import GraphModel
from Causal_Web.graph.io import load_graph, save_graph

def test_meta_node_roundtrip(tmp_path):
    data = {
        "nodes": {"A": {"x": 0, "y": 0}},
        "edges": [],
        "meta_nodes": {
            "MN1": {
                "members": ["A"],
                "constraints": {"phase_lock": {"tolerance": 0.1}},
                "type": "Configured",
                "collapsed": False,
                "x": 1.0,
                "y": 2.0,
            }
        },
    }
    path = tmp_path / "g.json"
    path.write_text(json.dumps(data))
    graph = load_graph(str(path))
    assert "MN1" in graph.meta_nodes
    out = tmp_path / "out.json"
    save_graph(str(out), graph)
    saved = json.loads(out.read_text())
    assert "meta_nodes" in saved
    assert saved["meta_nodes"]["MN1"]["members"] == ["A"]

