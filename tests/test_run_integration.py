import os
from Causal_Web.config import Config
from Causal_Web.graph.io import save_graph, load_graph
from Causal_Web.graph.model import GraphModel


def test_new_run_copies_graph(tmp_path, monkeypatch):
    input_dir = tmp_path / "input"
    runs_dir = tmp_path / "runs"
    input_dir.mkdir()
    runs_dir.mkdir()
    monkeypatch.setattr(Config, "input_dir", str(input_dir))
    monkeypatch.setattr(Config, "runs_dir", str(runs_dir))
    monkeypatch.setattr(Config, "config_file", str(input_dir / "config.json"))
    monkeypatch.setattr(Config, "graph_file", str(input_dir / "graph.json"))

    graph = GraphModel.blank(True)
    graph_path = input_dir / "graph.json"
    save_graph(str(graph_path), graph)
    (input_dir / "config.json").write_text("{}")

    run_dir = Config.new_run("test")
    copied = os.path.join(run_dir, "input", os.path.basename(graph_path))
    assert os.path.exists(copied)
    loaded = load_graph(copied)
    assert loaded.nodes
