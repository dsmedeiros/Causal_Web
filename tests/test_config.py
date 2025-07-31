import json
from copy import deepcopy
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


def test_logging_mode_filters_categories():
    original_mode = getattr(Config, "logging_mode", ["diagnostic"])
    original_files = deepcopy(Config.log_files)
    try:
        Config.logging_mode = ["tick"]
        Config.log_files = {
            "tick": {"coherence_log": True},
            "event": {"event_log": True},
        }
        assert Config.is_log_enabled("tick", "coherence_log")
        assert not Config.is_log_enabled("event", "event_log")
        assert Config.is_log_enabled("tick")
        assert not Config.is_log_enabled("event")
    finally:
        Config.logging_mode = original_mode
        Config.log_files = original_files


def test_load_smooth_phase_flag(tmp_path):
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"smooth_phase": True}))
    original = getattr(Config, "smooth_phase", False)
    Config.load_from_file(str(cfg))
    try:
        assert Config.smooth_phase is True
    finally:
        Config.smooth_phase = original


def test_load_propagation_flags(tmp_path):
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"propagation_control": {"enable_sip_child": False}}))
    original = Config.propagation_control.copy()
    Config.load_from_file(str(cfg))
    try:
        assert Config.propagation_control["enable_sip_child"] is False
    finally:
        Config.propagation_control = original
