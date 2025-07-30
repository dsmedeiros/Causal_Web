from Causal_Web.graph.model import GraphModel
import pytest


def _sample_nodes(model: GraphModel) -> None:
    model.nodes["A"] = {"x": 0, "y": 0}
    model.nodes["B"] = {"x": 1, "y": 1}


def test_add_edge():
    model = GraphModel.blank()
    _sample_nodes(model)
    model.add_connection("A", "B", delay=2, attenuation=0.5, connection_type="edge")
    assert model.edges[0]["from"] == "A"
    assert model.edges[0]["to"] == "B"
    assert model.edges[0]["delay"] == 2


def test_self_loop_controlled():
    model = GraphModel.blank()
    _sample_nodes(model)
    with pytest.raises(ValueError):
        model.add_connection("A", "A")
    model.nodes["A"]["allow_self_connection"] = True
    model.add_connection("A", "A")
    assert model.edges[0]["from"] == "A" and model.edges[0]["to"] == "A"


def test_duplicate_edge_disallowed():
    model = GraphModel.blank()
    _sample_nodes(model)
    model.add_connection("A", "B")
    with pytest.raises(ValueError):
        model.add_connection("A", "B")


def test_apply_spring_layout():
    model = GraphModel.blank()
    _sample_nodes(model)
    model.add_connection("A", "B")
    model.apply_spring_layout()
    assert "x" in model.nodes["A"]


def test_add_bridge_and_edit():
    model = GraphModel.blank()
    _sample_nodes(model)
    model.add_connection("A", "B", connection_type="bridge", attenuation=0.8)
    assert model.bridges
    model.update_connection(0, "bridge", delay=3)
    assert model.bridges[0]["delay"] == 3
    model.remove_connection(0, "bridge")
    assert not model.bridges
