import json
from Causal_Web.config import Config


def test_load_from_file_resolves_graph_file(tmp_path):
    cfg = tmp_path / "config.json"
    graph_path = tmp_path / "g.json"
    cfg.write_text(json.dumps({"graph_file": "g.json"}))
    original = Config.graph_file
    Config.load_from_file(str(cfg))
    try:
        assert Config.graph_file == str(graph_path)
    finally:
        Config.graph_file = original

