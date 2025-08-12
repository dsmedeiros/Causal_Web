import json
from pathlib import Path

from telemetry.metrics import MetricsLogger


def test_metrics_logger(tmp_path: Path):
    logger = MetricsLogger(tmp_path)
    logger.log(0, {"g": 1.0}, {"Delta": 1.0}, 0, {}, {})
    logger.flush(
        type("Cfg", (), {"samples": 1, "groups": {"g": (0, 1)}, "seed": 0})(),
        [{"g": 1.0}],
    )
    assert (tmp_path / "metrics.csv").exists()
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["seed"] == 0
