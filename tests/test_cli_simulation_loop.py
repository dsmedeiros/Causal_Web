import sys
from Causal_Web.main import MainService
from Causal_Web.config import Config
from Causal_Web.engine.engine_v2 import adapter


def test_cli_headless_uses_v2_simulation_loop(monkeypatch, tmp_path):
    """CLI should invoke engine_v2.adapter.simulation_loop without GUI worker."""
    cfg = tmp_path / "c.json"
    cfg.write_text("{}")

    called = {}

    def fake_build_graph(path=None):
        called["build"] = True

    def fake_loop():
        called["loop"] = True
        with Config.state_lock:
            Config.is_running = False

    monkeypatch.setattr(adapter, "build_graph", fake_build_graph)
    monkeypatch.setattr(adapter, "simulation_loop", fake_loop)

    service = MainService(argv=["--config", str(cfg), "--no-gui"])
    service.run()

    assert called.get("loop")
    assert "Causal_Web.gui_legacy.engine_worker" not in sys.modules
