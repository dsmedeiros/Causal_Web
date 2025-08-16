import asyncio

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
