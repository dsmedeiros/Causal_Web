import json
import pytest

pytest.importorskip("PySide6")
from ui_new.state.MCTS import MCTSModel
from experiments.ablation import AblationResult


def test_local_ablation_writes_file(tmp_path, monkeypatch):
    model = MCTSModel()
    monkeypatch.setattr(
        "ui_new.state.MCTS.load_top_k",
        lambda path: {"rows": [{"groups": {"a": 1.0, "b": 2.0}}]},
    )
    monkeypatch.setattr("ui_new.state.MCTS.run_gates", lambda raw, gates, frames=1: {})
    calls = []

    def fake_local(best, evaluate, features, span=0.1, steps=20):
        calls.append(features)
        return AblationResult(features, [[best[f] for f in features]], [[0.0]])

    monkeypatch.setattr("ui_new.state.MCTS.local_ablation", fake_local)
    monkeypatch.chdir(tmp_path)
    model.localAblation()
    out = tmp_path / "experiments" / "local_ablation.json"
    data = json.loads(out.read_text())
    assert len(calls) == 3
    assert data["ablations"][0]["params"] == calls[0]
    assert model.ablations[0]["params"] == calls[0]
