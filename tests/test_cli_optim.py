import json
from pathlib import Path

import yaml

from experiments import MCTS_H
from cw.cli import main


def _base_config():
    return {
        "W0": 1.0,
        "Delta": 1.0,
        "alpha_leak": 1.0,
        "alpha_d": 1.0,
        "lambda_decay": 1.0,
        "sigma_reinforce": 1.0,
        "a": 1.0,
        "b": 1.0,
        "eta": 1.0,
    }


def test_cli_mcts_h(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path("base.yaml").write_text(yaml.safe_dump(_base_config()))
    Path("space.yaml").write_text(yaml.safe_dump(["Delta_over_W0"]))
    topk = {"rows": [{"groups": {"Delta_over_W0": 0.2}}]}
    Path("topk.json").write_text(json.dumps(topk))

    frames_seen = []

    def fake_run_gates(raw, gates, frames=None):
        frames_seen.append(frames)
        return {"G6_CHSH": 0.0, "target_success": 0.0, "coherence": 0.0}

    monkeypatch.setattr("experiments.queue.run_gates", fake_run_gates)
    monkeypatch.setattr(
        "experiments.queue.checks.from_metrics",
        lambda metrics: {
            "inv_conservation_residual": 0.0,
            "inv_no_signaling_delta": 0.0,
            "inv_causality_ok": True,
            "inv_ancestry_ok": True,
        },
    )

    args = [
        "optim",
        "--optim",
        "mcts_h",
        "--base",
        "base.yaml",
        "--space",
        "space.yaml",
        "--priors",
        "topk.json",
        "--budget",
        "1",
        "--promote-threshold",
        "1.0",
        "--state",
        "state.json",
        "--proxy-frames",
        "10",
        "--full-frames",
        "20",
    ]
    main(args)
    assert frames_seen == [10]
    assert Path("state.json").exists()
    assert not Path("experiments/top_k.json").exists()
    main(args)
    data = json.loads(Path("experiments/top_k.json").read_text())
    assert len(data["rows"]) == 1
    assert frames_seen == [10, 20]


def test_cli_mcts_h_promote_window(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path("base.yaml").write_text(yaml.safe_dump(_base_config()))
    Path("space.yaml").write_text(yaml.safe_dump(["Delta_over_W0"]))
    captured: dict[str, int] = {}

    real_init = MCTS_H.__init__

    def fake_init(self, space, priors, cfg, rng=None):  # type: ignore[override]
        captured["promote_window"] = cfg.get("promote_window")
        real_init(self, space, priors, cfg, rng)

    monkeypatch.setattr(MCTS_H, "__init__", fake_init)
    args = [
        "optim",
        "--optim",
        "mcts_h",
        "--base",
        "base.yaml",
        "--space",
        "space.yaml",
        "--budget",
        "0",
        "--promote-window",
        "5",
    ]
    main(args)
    assert captured["promote_window"] == 5


def test_cli_mcts_h_quantile(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path("base.yaml").write_text(yaml.safe_dump(_base_config()))
    Path("space.yaml").write_text(yaml.safe_dump(["Delta_over_W0"]))
    topk = {"rows": [{"groups": {"Delta_over_W0": 0.2}}]}
    Path("topk.json").write_text(json.dumps(topk))

    frames_seen = []

    def fake_run_gates(raw, gates, frames=None):
        frames_seen.append(frames)
        return {"G6_CHSH": 0.0, "target_success": 0.0, "coherence": 0.0}

    monkeypatch.setattr("experiments.queue.run_gates", fake_run_gates)
    monkeypatch.setattr(
        "experiments.queue.checks.from_metrics",
        lambda metrics: {
            "inv_conservation_residual": 0.0,
            "inv_no_signaling_delta": 0.0,
            "inv_causality_ok": True,
            "inv_ancestry_ok": True,
        },
    )

    args = [
        "optim",
        "--optim",
        "mcts_h",
        "--base",
        "base.yaml",
        "--space",
        "space.yaml",
        "--priors",
        "topk.json",
        "--budget",
        "1",
        "--promote-quantile",
        "1.0",
        "--bins",
        "2",
        "--state",
        "state.json",
        "--proxy-frames",
        "10",
        "--full-frames",
        "20",
    ]
    main(args)
    assert frames_seen == [10]
    assert Path("state.json").exists()
    assert not Path("experiments/top_k.json").exists()
    main(args)
    data = json.loads(Path("experiments/top_k.json").read_text())
    assert len(data["rows"]) == 1
    assert frames_seen == [10, 20]


def test_cli_mcts_h_multi_objective(tmp_path, monkeypatch):
    """CLI flag enables multi-objective mode in optimiser config."""

    monkeypatch.chdir(tmp_path)
    Path("base.yaml").write_text(yaml.safe_dump(_base_config()))
    Path("space.yaml").write_text(yaml.safe_dump(["Delta_over_W0"]))

    captured: dict[str, bool] = {}
    real_init = MCTS_H.__init__

    def fake_init(self, space, priors, cfg, rng=None):  # type: ignore[override]
        captured["multi_objective"] = cfg.get("multi_objective", False)
        real_init(self, space, priors, cfg, rng)

    monkeypatch.setattr(MCTS_H, "__init__", fake_init)

    args = [
        "optim",
        "--optim",
        "mcts_h",
        "--base",
        "base.yaml",
        "--space",
        "space.yaml",
        "--budget",
        "0",
        "--multi-objective",
    ]
    main(args)
    assert captured["multi_objective"]
