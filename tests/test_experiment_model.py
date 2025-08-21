import asyncio
import pytest

pytest.importorskip("PySide6")
from ui_new.state.Experiment import ExperimentModel


class DummyClient:
    def __init__(self) -> None:
        self.sent: list[dict] = []

    async def send(self, msg) -> None:
        self.sent.append(msg)


def test_step_and_rate(monkeypatch):
    model = ExperimentModel()
    client = DummyClient()
    model.set_client(client)

    monkeypatch.setattr(asyncio, "create_task", lambda c: asyncio.run(c))

    model.step()
    model.setRate(0.5)

    assert {"ExperimentControl": {"action": "step"}} in client.sent
    assert {"ExperimentControl": {"action": "set_rate", "rate": 0.5}} in client.sent


def test_run_baseline(monkeypatch, tmp_path):
    model = ExperimentModel()
    client = DummyClient()
    model.set_client(client)

    monkeypatch.chdir(tmp_path)
    cfg_dir = tmp_path / "experiments"
    cfg_dir.mkdir()
    cfg_path = cfg_dir / "best_config.yaml"
    cfg_path.write_text("dimensionless:\n  Delta_over_W0: 0.25\nseed: 7\n")
    monkeypatch.setattr(asyncio, "create_task", lambda c: asyncio.run(c))

    model.runBaseline()

    assert client.sent
    msg = client.sent[0]["ExperimentControl"]
    assert msg["action"] == "run"
    assert msg["config"]["seed"] == 7
    assert abs(msg["config"]["Delta"] - 0.25) < 1e-6
