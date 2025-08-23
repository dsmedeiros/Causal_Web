import json
from pathlib import Path

import pytest

from experiments import OptimizerQueueManager, MCTS_H
from experiments.reproduce import reproduce_run


def test_reproduce_mcts_run(tmp_path, monkeypatch):
    """Re-running a persisted MCTS run yields identical results."""

    from Causal_Web.config import Config

    monkeypatch.chdir(tmp_path)
    Config.output_dir = str(tmp_path)
    base = {}
    opt = MCTS_H(["prob"], {}, {"rng_seed": 0, "promote_threshold": 0.0})

    def fitness(metrics, inv, groups):
        return metrics["G1"]

    mgr = OptimizerQueueManager(base, [1], fitness, opt, proxy_frames=1, full_frames=1)
    run_dir: Path | None = None
    for _ in range(5):
        res = mgr.run_next()
        assert res is not None
        if res.status == "full" and res.path:
            run_dir = Path("experiments") / res.path
            break
    assert run_dir is not None

    reproduced = reproduce_run(run_dir)
    recorded = json.loads((run_dir / "result.json").read_text())
    for k, v in recorded["metrics"].items():
        assert reproduced["metrics"][k] == pytest.approx(v)
    for k, v in recorded["invariants"].items():
        assert reproduced["invariants"][k] == pytest.approx(v)
    state = json.loads(Path("experiments/mcts_state.json").read_text())
    assert reproduced["tree_hash"] == state["tree_hash"]
