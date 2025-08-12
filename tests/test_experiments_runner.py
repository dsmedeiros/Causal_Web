import json
from pathlib import Path

from experiments.runner import run


def create_config(tmp_path: Path) -> Path:
    text = """
samples = 2
seed = 42
gates = [1]
[groups]
Delta_over_W0 = [0.5, 1.0]
alpha_d_over_leak = [1.0, 2.0]
"""
    path = tmp_path / "exp.toml"
    path.write_text(text)
    return path


def test_runner_deterministic(tmp_path: Path):
    cfg_path = create_config(tmp_path)
    out_dir = tmp_path / "run"
    run(cfg_path, out_dir)
    summary1 = json.loads((out_dir / "summary.json").read_text())
    metrics1 = (out_dir / "metrics.csv").read_text()

    out_dir2 = tmp_path / "run2"
    run(cfg_path, out_dir2)
    summary2 = json.loads((out_dir2 / "summary.json").read_text())
    metrics2 = (out_dir2 / "metrics.csv").read_text()

    assert summary1 == summary2
    assert metrics1 == metrics2
