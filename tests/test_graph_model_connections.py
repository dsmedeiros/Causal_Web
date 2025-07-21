from Causal_Web.graph.model import GraphModel


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


def test_add_bridge_and_edit():
    model = GraphModel.blank()
    _sample_nodes(model)
    model.add_connection("A", "B", connection_type="bridge", attenuation=0.8)
    assert model.bridges
    model.update_connection(0, "bridge", delay=3)
    assert model.bridges[0]["delay"] == 3
    model.remove_connection(0, "bridge")
    assert not model.bridges
