import json
from Causal_Web.config import Config


def test_save_log_files_updates_config(tmp_path, monkeypatch):
    path = tmp_path / "config.json"
    data = {"log_files": {"tick": {"a": True}, "event": {"b": False}}}
    path.write_text(json.dumps(data))
    monkeypatch.setattr(Config, "config_file", str(path))
    Config.log_files = {"tick": {"a": False}, "event": {"b": True}}
    Config.save_log_files()
    saved = json.loads(path.read_text())
    assert saved["log_files"] == Config.log_files
