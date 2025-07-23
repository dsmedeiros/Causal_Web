import pytest
from Causal_Web.graph.model import GraphModel


def test_remove_node_cleans_references():
    model = GraphModel.blank()
    model.add_node("A")
    model.add_node("B")
    model.add_connection("A", "B")
    model.add_observer("OBS1", monitors=["A", "B"], x=0.0, y=0.0)
    model.add_meta_node("MN1", members=["A", "B"], x=0.0, y=0.0)

    model.remove_node("B")

    assert "B" not in model.nodes
    assert all("B" not in (e["from"], e["to"]) for e in model.edges)
    assert "B" not in model.meta_nodes["MN1"]["members"]
    assert "B" not in model.observers[0]["monitors"]


def test_remove_observer():
    model = GraphModel.blank()
    model.add_observer("O1")
    model.add_observer("O2")
    model.remove_observer(0)
    assert len(model.observers) == 1
    assert model.observers[0]["id"] == "O2"


def test_remove_meta_node():
    model = GraphModel.blank()
    model.add_meta_node("MN1")
    model.add_meta_node("MN2")
    model.remove_meta_node("MN1")
    assert "MN1" not in model.meta_nodes
    assert "MN2" in model.meta_nodes
