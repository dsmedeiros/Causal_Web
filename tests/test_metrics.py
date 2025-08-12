import csv
import json
from pathlib import Path

from telemetry.metrics import MetricsLogger


def test_metrics_logger(tmp_path: Path):
    logger = MetricsLogger(tmp_path)
    logger.log(0, {"g": 1.0}, {"Delta": 1.0}, 0, {"G1": 1.0}, {})
    logger.log(1, {"g": 2.0}, {"Delta": 2.0}, 1, {"G2": 2.0}, {})
    cfg = type(
        "Cfg",
        (),
        {"samples": 2, "groups": {"g": (0, 1)}, "seed": 0, "gates": [1, 2]},
    )()
    logger.flush(cfg, [{"g": 1.0}, {"g": 2.0}])
    csv_path = tmp_path / "metrics.csv"
    assert csv_path.exists()
    rows = list(csv.DictReader(csv_path.open()))
    assert rows[1]["G2"] == "2.0"
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["seed"] == 0
    assert summary["gates"] == [1, 2]
    assert summary["metrics_agg"]["mean_G1"] == 1.0
    assert summary["metrics_agg"]["std_G1"] == 0.0
    assert summary["metrics_agg"]["mean_G2"] == 2.0
