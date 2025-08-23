import asyncio

import pytest

from ui_new.state.MCTS import MCTSModel
from experiments.artifacts import TopKEntry, update_top_k


def test_mcts_model_promote(tmp_path, monkeypatch):
    exp_dir = tmp_path / "experiments"
    exp_dir.mkdir()
    entry = TopKEntry(
        run_id="r1",
        fitness=0.0,
        objectives={"f0": 0.0},
        groups={"Delta_over_W0": 0.2},
        toggles={},
        seed=1,
        path="r1",
    )
    update_top_k([entry], exp_dir / "top_k.json")
    monkeypatch.chdir(tmp_path)
    model = MCTSModel()
    model.promoteBaseline()
    assert (exp_dir / "best_config.yaml").exists()


def test_mcts_model_metrics(tmp_path, monkeypatch):
    exp_dir = tmp_path / "experiments"
    exp_dir.mkdir()
    monkeypatch.chdir(tmp_path)
    model = MCTSModel()
    model.promoteThreshold = 1.0
    model.maxNodes = 100
    model.proxyFrames = 5
    model.fullFrames = 5

    async def run_model() -> None:
        model.start()
        for _ in range(50):
            await asyncio.sleep(0)
            if (
                model.proxyEvaluations
                and model.proxyEvaluations == model.fullEvaluations
            ):
                break
        model.pause()
        await asyncio.sleep(0)

    asyncio.run(run_model())

    assert model.nodeCount > 0
    assert model.proxyEvaluations == model.fullEvaluations > 0
    assert model.promotionRate == pytest.approx(1.0)
    assert model.frontier >= 0
    assert 0 <= model.expansionRate <= 1
