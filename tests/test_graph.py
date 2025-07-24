import math
from Causal_Web.engine.graph import CausalGraph
from Causal_Web.engine.node import Node
from Causal_Web.engine.node import Edge
from Causal_Web.config import Config
import json


def test_interference_type_constructive():
    g = CausalGraph()
    phases = [0.0, 0.05]
    assert g._interference_type(phases) == "constructive"


def test_interference_type_destructive():
    g = CausalGraph()
    phases = [0.0, math.pi]
    assert g._interference_type(phases) == "destructive"


def test_detect_clusters():
    g = CausalGraph()
    for nid, freq in {"A": 1.0, "B": 1.02, "C": 1.5}.items():
        g.nodes[nid] = Node(nid)
        g.nodes[nid].law_wave_frequency = freq
        g.nodes[nid].coherence = 0.95
    clusters = g.detect_clusters(coherence_threshold=0.9, freq_tolerance=0.05)
    assert [sorted(c) for c in clusters] == [["A", "B"]]


def test_edge_adjusted_delay():
    edge = Edge("A", "B", attenuation=1.0, density=0.5, delay=2)
    # with zero freq difference sin term is zero -> delay = delay + int(density)
    assert edge.adjusted_delay(1.0, 1.0, kappa=1.0) == 2 + int(0.5)
    # with frequency difference effect
    delay = edge.adjusted_delay(1.0, 1.5, kappa=1.0)
    assert delay >= 1


def test_remove_node_cleans_references():
    g = CausalGraph()
    g.add_node("A")
    g.add_node("B")
    g.add_edge("A", "B")

    g.remove_node("B")

    assert "B" not in g.nodes
    assert "B" not in g.edges_from
    assert "B" not in g.edges_to
    assert all("B" not in (e.source, e.target) for e in g.edges)

    # should not raise when computing clusters
    g.hierarchical_clusters()


def test_load_from_file_edges_dict(tmp_path):
    path = tmp_path / "g.json"
    data = {"nodes": {"A": {}, "B": {}}, "edges": {"A": ["B"]}}
    path.write_text(json.dumps(data))

    g = CausalGraph()
    g.load_from_file(str(path))

    edges = g.get_edges_from("A")
    assert len(edges) == 1
    assert edges[0].target == "B"


def test_edge_weights_affect_delay(monkeypatch):
    old_range = getattr(Config, "edge_weight_range", [1.0, 1.0])
    Config.edge_weight_range = [2.0, 2.0]
    g = CausalGraph()
    g.add_node("A")
    g.add_node("B")
    g.add_edge("A", "B")
    edge = g.get_edges_from("A")[0]
    assert edge.weight == 2.0
    delay = edge.adjusted_delay(1.0, 1.0, kappa=1.0)
    assert delay >= 2
    Config.edge_weight_range = old_range


def test_load_from_file_resets_spatial_index(tmp_path):
    g = CausalGraph()
    path1 = tmp_path / "g1.json"
    data1 = {"nodes": {"A": {"x": 0, "y": 0}, "B": {"x": 50, "y": 0}}}
    path1.write_text(json.dumps(data1))
    g.load_from_file(str(path1))
    assert {nid for ids in g.spatial_index.values() for nid in ids} == {"A", "B"}

    path2 = tmp_path / "g2.json"
    data2 = {"nodes": {"C": {"x": 0, "y": 0}}}
    path2.write_text(json.dumps(data2))
    g.load_from_file(str(path2))
    assert {nid for ids in g.spatial_index.values() for nid in ids} == {"C"}
