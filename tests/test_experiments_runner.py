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


def create_base(tmp_path: Path) -> Path:
    base = {
        "W0": 1.0,
        "Delta": 1.0,
        "alpha_d": 1.0,
        "alpha_leak": 1.0,
    }
    path = tmp_path / "base.yaml"
    import yaml

    path.write_text(yaml.safe_dump(base))
    return path


def test_runner_deterministic(tmp_path: Path):
    cfg_path = create_config(tmp_path)
    base_path = create_base(tmp_path)
    out_dir = tmp_path / "run"
    run(cfg_path, base_path, out_dir)
    summary1 = json.loads((out_dir / "summary.json").read_text())
    import csv

    with (out_dir / "metrics.csv").open() as fh:
        metrics1 = list(csv.DictReader(fh))
        for row in metrics1:
            row.pop("ts", None)

    out_dir2 = tmp_path / "run2"
    run(cfg_path, base_path, out_dir2)
    summary2 = json.loads((out_dir2 / "summary.json").read_text())
    with (out_dir2 / "metrics.csv").open() as fh:
        metrics2 = list(csv.DictReader(fh))
        for row in metrics2:
            row.pop("ts", None)

    assert summary1 == summary2
    assert metrics1 == metrics2


def test_metrics_csv_has_gate_and_invariants(tmp_path: Path):
    cfg_path = create_config(tmp_path)
    base_path = create_base(tmp_path)
    out_dir = tmp_path / "run"
    run(cfg_path, base_path, out_dir)
    import csv

    with (out_dir / "metrics.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    assert "G1_visibility" in rows[0]
    assert "inv_causality_ok" in rows[0]
    assert "inv_conservation_residual" in rows[0]
    assert "inv_no_signaling_delta" in rows[0]
    assert "inv_ancestry_ok" in rows[0]
    assert "inv_gate_determinism_ok" in rows[0]


def test_runner_process_pool_determinism(tmp_path: Path):
    cfg_path = create_config(tmp_path)
    base_path = create_base(tmp_path)
    out1 = tmp_path / "p1"
    run(cfg_path, base_path, out1, parallel=1)
    import csv

    with (out1 / "metrics.csv").open() as fh:
        metrics1 = list(csv.DictReader(fh))
        for row in metrics1:
            row.pop("ts", None)

    out2 = tmp_path / "p8"
    run(cfg_path, base_path, out2, parallel=8, use_processes=True)
    with (out2 / "metrics.csv").open() as fh:
        metrics2 = list(csv.DictReader(fh))
        for row in metrics2:
            row.pop("ts", None)

    assert metrics1 == metrics2
