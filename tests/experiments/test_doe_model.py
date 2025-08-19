import json
from pathlib import Path

import yaml

from ui_new.state.DOE import DOEModel


def test_doe_promote_uses_run_config(tmp_path, monkeypatch):
    run_dir = tmp_path / "experiments" / "runs" / "2025-01-01" / "abc"
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(json.dumps({"x": 1.0}))
    monkeypatch.chdir(tmp_path)
    model = DOEModel()
    model._topk = [{"fitness": 0.0, "path": "runs/2025-01-01/abc"}]
    model.promote()
    data = yaml.safe_load((Path("experiments/best_config.yaml")).read_text())
    assert data == {"x": 1.0}


def test_doe_topk_paths_link_runs(tmp_path, monkeypatch) -> None:
    from Causal_Web.config import Config

    monkeypatch.chdir(tmp_path)
    Config.output_dir = str(tmp_path)
    (tmp_path / "delta_log.jsonl").write_text("{}\n")
    model = DOEModel()
    model.runLhs(1)
    data = json.loads((tmp_path / "experiments/top_k.json").read_text())
    row = data["rows"][0]
    run_dir = tmp_path / "experiments" / row["path"]
    assert (run_dir / "config.json").exists()
    assert (run_dir / "result.json").exists()
    assert (run_dir / "delta_log.jsonl").exists()


def test_doe_persists_run_once(tmp_path, monkeypatch) -> None:
    from Causal_Web.config import Config

    monkeypatch.chdir(tmp_path)
    Config.output_dir = str(tmp_path)
    (tmp_path / "delta_log.jsonl").write_text("{}\n")
    model = DOEModel()
    model.runLhs(1)
    data = json.loads((tmp_path / "experiments/top_k.json").read_text())
    path1 = data["rows"][0]["path"]
    # Trigger a second recompute and ensure the path stays the same
    model._recompute()
    data = json.loads((tmp_path / "experiments/top_k.json").read_text())
    path2 = data["rows"][0]["path"]
    assert path1 == path2
