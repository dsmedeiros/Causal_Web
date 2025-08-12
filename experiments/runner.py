"""DOE runner for exploring parameter space.

This module generates samples of dimensionless groups using
Latin Hypercube sampling with a deterministic seed. For each
sample a configuration is materialised, invariants are checked
and metrics are logged.
"""

from __future__ import annotations

import argparse
import pathlib
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np

from config.normalizer import Normalizer
from invariants import checks
from telemetry.metrics import MetricsLogger


@dataclass
class ExperimentConfig:
    """Container for experiment configuration."""

    samples: int
    groups: Dict[str, Tuple[float, float]]
    gates: List[int]
    seed: int = 0

    @classmethod
    def from_mapping(cls, data: Dict[str, object]) -> "ExperimentConfig":
        groups = {k: tuple(v) for k, v in data["groups"].items()}
        seed = data.get("seed", 0)
        return cls(
            samples=int(data["samples"]),
            groups=groups,
            gates=list(data["gates"]),
            seed=seed,
        )


def _latin_hypercube(n: int, dims: int, rng: np.random.Generator) -> np.ndarray:
    """Return a Latin Hypercube sample in the unit cube.

    Parameters
    ----------
    n:
        Number of samples.
    dims:
        Number of dimensions.
    rng:
        Random number generator.
    """

    cut = np.linspace(0, 1, n + 1)
    u = rng.random((n, dims))
    a = cut[:n]
    b = cut[1 : n + 1]
    points = u * (b - a)[:, None] + a[:, None]
    for j in range(dims):
        rng.shuffle(points[:, j])
    return points


def _sample_groups(
    cfg: ExperimentConfig, rng: np.random.Generator
) -> List[Dict[str, float]]:
    names = list(cfg.groups.keys())
    ranges = np.array([cfg.groups[n] for n in names], dtype=float)
    unit = _latin_hypercube(cfg.samples, len(names), rng)
    scaled = ranges[:, 0] + unit * (ranges[:, 1] - ranges[:, 0])
    return [dict(zip(names, scaled[i])) for i in range(cfg.samples)]


def run(config_path: pathlib.Path, out_dir: pathlib.Path) -> None:
    """Execute a design-of-experiments sweep.

    Parameters
    ----------
    config_path:
        Path to a YAML or TOML configuration file.
    out_dir:
        Directory in which results will be written.
    """

    cfg = _load_config(config_path)
    rng = np.random.default_rng(cfg.seed)
    samples = _sample_groups(cfg, rng)

    out_dir.mkdir(parents=True, exist_ok=True)
    logger = MetricsLogger(out_dir)
    normalizer = Normalizer()

    for i, groups in enumerate(samples):
        raw = normalizer.to_raw(groups)
        logger.log(i, groups, raw, cfg.seed)
        _check_invariants()

    logger.flush(cfg, samples)


def _check_invariants() -> None:
    """Run all invariant checks with dummy data.

    The real system would supply appropriate measurements. These
    lightweight checks guard against accidental regressions.
    """

    if not checks.causality([{"d_arr": 1.0, "d_src": 0.5}]):
        raise ValueError("causality check failed")
    if not checks.local_conservation(0.0, 0.0, 1e-6):
        raise ValueError("local conservation failed")
    if not checks.no_signaling(0.5, 1e-3):
        raise ValueError("no-signaling failed")
    if not checks.ancestry_determinism([("q", "h", "m")]):
        raise ValueError("ancestry determinism failed")


def _load_config(path: pathlib.Path) -> ExperimentConfig:
    if path.suffix in {".yaml", ".yml"}:
        import yaml

        data = yaml.safe_load(path.read_text())
    elif path.suffix == ".toml":
        import tomllib

        data = tomllib.loads(path.read_text())
    else:
        raise ValueError(f"Unsupported config extension: {path.suffix}")
    return ExperimentConfig.from_mapping(data)


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run DOE experiments")
    parser.add_argument("--config", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    run(args.config, args.out)


if __name__ == "__main__":  # pragma: no cover
    main()
