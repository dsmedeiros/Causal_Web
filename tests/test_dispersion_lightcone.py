from pathlib import Path

from Causal_Web.config import Config
from experiments.dispersion import run_dispersion, compute_dispersion
from experiments.lightcone import run_lightcone, simulate_lightcone


def test_dispersion_runs(tmp_path: Path):
    Config.qwalk["enabled"] = True
    Config.qwalk["thetas"] = {"theta1": 0.35, "theta2": 0.2}
    Config.dispersion["k_values"] = [0.0, 0.1, 0.2]
    out = tmp_path / "dispersion.csv"
    rows = run_dispersion(str(out))
    assert out.exists()
    assert len(rows) == 3
    assert rows[1]["omega"] > rows[0]["omega"]


def test_lightcone_runs(tmp_path: Path):
    Config.qwalk["max_distance"] = 5
    out = tmp_path / "lightcone.csv"
    rows = run_lightcone(str(out))
    assert out.exists()
    assert rows[-1]["arrival_depth"] >= rows[0]["arrival_depth"]
    assert rows[-1]["distance"] == 5
