import math
from Causal_Web.engine.graph import CausalGraph
from Causal_Web.engine.node import Node
from Causal_Web.engine.node import Edge


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
    for nid, freq in {"A":1.0, "B":1.02, "C":1.5}.items():
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
