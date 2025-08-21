"""DOE runner for exploring parameter space.

This module generates samples of dimensionless groups using
Latin Hypercube sampling with a deterministic seed. For each
sample a configuration is materialised, invariants are checked
and metrics are logged.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import logging
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Tuple

import numpy as np

from config.normalizer import Normalizer
from invariants import checks
from telemetry.metrics import MetricsLogger
from .gates import run_gates
from .index import RunIndex, run_key
from .artifacts import persist_run, allocate_run_dir


@dataclass
class ExperimentConfig:
    """Container for experiment configuration."""

    samples: int
    groups: Dict[str, Tuple[float, float]]
    gates: List[int]
    seed: int = 0
    tol: Dict[str, float] = field(
        default_factory=lambda: {"leak": 1e-6, "marginal": 0.02}
    )

    @classmethod
    def from_mapping(cls, data: Dict[str, object]) -> "ExperimentConfig":
        """Construct an :class:`ExperimentConfig` from a generic mapping.

        Parameters
        ----------
        data:
            Mapping containing configuration fields.

        Raises
        ------
        KeyError
            If required configuration keys are missing.
        """

        required = {"samples", "groups", "gates"}
        missing = required - data.keys()
        if missing:
            raise KeyError(
                f"Experiment configuration missing keys: {', '.join(sorted(missing))}"
            )

        groups = {k: tuple(v) for k, v in data["groups"].items()}
        seed = data.get("seed", 0)
        tol = {k: float(v) for k, v in data.get("tol", {}).items()}
        return cls(
            samples=int(data["samples"]),
            groups=groups,
            gates=list(data["gates"]),
            seed=seed,
            tol={"leak": 1e-6, "marginal": 0.02} | tol,
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
    lows, highs = ranges[:, 0], ranges[:, 1]
    scaled = lows[None, :] + unit * (highs - lows)[None, :]
    return [dict(zip(names, scaled[i])) for i in range(cfg.samples)]


def _mix(seed: int, i: int) -> int:
    x = (seed ^ (i + 0x9E3779B9)) & 0xFFFFFFFF
    x ^= x >> 16
    x = (x * 0x7FEB352D) & 0xFFFFFFFF
    x ^= x >> 15
    x = (x * 0x846CA68B) & 0xFFFFFFFF
    return x


def _write_yaml(path: pathlib.Path, data: Dict[str, float]) -> None:
    import yaml
    import numpy as np

    clean = {k: float(v) if isinstance(v, np.floating) else v for k, v in data.items()}
    path.write_text(yaml.safe_dump(clean))


def _process_sample(
    i: int,
    groups: Dict[str, float],
    cfg: ExperimentConfig,
    normalizer: Normalizer,
    base: Dict[str, float],
    out_dir: pathlib.Path,
) -> Tuple[
    int, Dict[str, float], Dict[str, float], int, Dict[str, float], Dict[str, float]
]:
    """Materialise a sample, execute gates and return metrics."""

    sample_seed = _mix(cfg.seed, i)
    raw = normalizer.to_raw(base, groups)
    raw["seed"] = sample_seed
    _write_yaml(out_dir / f"cfg_{i:04d}.yaml", raw)
    gate_metrics = run_gates(raw, cfg.gates)
    inv = checks.from_metrics(gate_metrics)
    if not inv["inv_causality_ok"]:
        raise ValueError("causality check failed")
    if abs(inv["inv_conservation_residual"]) > cfg.tol["leak"]:
        raise ValueError("local conservation failed")
    if abs(inv["inv_no_signaling_delta"]) > cfg.tol["marginal"]:
        raise ValueError("no-signaling failed")
    if not inv["inv_ancestry_ok"]:
        raise ValueError("ancestry determinism failed")
    return i, groups, raw, sample_seed, gate_metrics, inv


def run(
    exp_path: pathlib.Path,
    base_path: pathlib.Path,
    out_dir: pathlib.Path,
    parallel: int = 1,
    use_processes: bool = False,
    *,
    force: bool = False,
) -> None:
    """Execute a design-of-experiments sweep.

    Parameters
    ----------
    exp_path, base_path, out_dir:
        Paths to the experiment configuration, base configuration and output
        directory.
    parallel:
        Number of parallel workers. Uses a thread pool by default.
    use_processes:
        If ``True`` and ``parallel > 1``, a ``ProcessPoolExecutor`` is used
        instead of a thread pool. This helps when gate code is Python-heavy
        while preserving determinism via per-sample seeds.
    force:
        When ``True`` evaluate configurations even if present in the run index.

    Notes
    -----
    Parameters are sampled from dimensionless groups and converted to raw
    configurations before the experiment is executed. Results are logged in
    a deterministic order even when processed in parallel. If at least two
    samples are collected the standard deviation of each gate metric is
    analysed and a warning is emitted when any metric shows zero variation.
    """

    cfg = _load_config(exp_path)
    rng = np.random.default_rng(cfg.seed)
    samples = _sample_groups(cfg, rng)

    out_dir.mkdir(parents=True, exist_ok=True)
    logger = MetricsLogger(out_dir)
    normalizer = Normalizer()
    base = _load_base_config(base_path)
    index = RunIndex()

    results: List[
        Tuple[
            int,
            Dict[str, float],
            Dict[str, float],
            int,
            Dict[str, float],
            Dict[str, float],
        ]
    ] = []

    pending: List[Tuple[int, Dict[str, float], int, str]] = []
    for i, g in enumerate(samples):
        seed = _mix(cfg.seed, i)
        key = run_key({"groups": g, "toggles": {}, "seed": seed, "gates": cfg.gates})
        info = index.get(key)
        if info is not None and not force:
            run_dir = index.runs_root / info
            try:
                raw = json.loads((run_dir / "config.json").read_text())
            except Exception:
                raw = normalizer.to_raw(base, g)
            try:
                res = json.loads((run_dir / "result.json").read_text())
            except Exception:
                res = {"metrics": {}, "invariants": {}}
            results.append(
                (
                    i,
                    g,
                    raw,
                    seed,
                    res.get("metrics", {}),
                    res.get("invariants", {}),
                )
            )
            continue
        pending.append((i, g, seed, key))

    if parallel > 1 and pending:
        executor_cls = ProcessPoolExecutor if use_processes else ThreadPoolExecutor
        with executor_cls(max_workers=parallel) as ex:
            futs = [
                ex.submit(_process_sample, i, g, cfg, normalizer, base, out_dir)
                for i, g, _, _ in pending
            ]
            for (i, g, seed, key), fut in zip(pending, futs):
                result = fut.result()
                _, _, raw, _, gm, inv = result
                rid, abs_path, rel = allocate_run_dir()
                manifest = {
                    "run_id": rid,
                    "run_key": key,
                    "groups": g,
                    "toggles": {},
                    "seed": seed,
                    "gates": cfg.gates,
                }
                persist_run(
                    raw, {"metrics": gm, "invariants": inv}, abs_path, manifest=manifest
                )
                index.mark(key, rel)
                results.append((i, g, raw, seed, gm, inv))
    else:
        for i, g, seed, key in pending:
            res = _process_sample(i, g, cfg, normalizer, base, out_dir)
            _, _, raw, _, gm, inv = res
            rid, abs_path, rel = allocate_run_dir()
            manifest = {
                "run_id": rid,
                "run_key": key,
                "groups": g,
                "toggles": {},
                "seed": seed,
                "gates": cfg.gates,
            }
            persist_run(
                raw, {"metrics": gm, "invariants": inv}, abs_path, manifest=manifest
            )
            index.mark(key, rel)
            results.append((i, g, raw, seed, gm, inv))

    for i, groups, raw, seed, gm, inv in sorted(results, key=lambda x: x[0]):
        logger.log(i, groups, raw, seed, gm, inv)
    if len(results) >= 2:
        metric_vals: Dict[str, List[float]] = {}
        for _, _, _, _, gm, _ in results:
            for k, v in gm.items():
                if isinstance(v, (int, float)):
                    metric_vals.setdefault(k, []).append(float(v))
        if any(np.std(vals) == 0.0 for vals in metric_vals.values()):
            logging.warning(
                "no variation across DOE — check gate wiring/groups mapping."
            )

    logger.flush(cfg, samples)


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


def _load_base_config(path: pathlib.Path) -> Dict[str, float]:
    if path.suffix in {".yaml", ".yml"}:
        import yaml

        return yaml.safe_load(path.read_text())
    elif path.suffix == ".toml":
        import tomllib

        return tomllib.loads(path.read_text())
    else:
        raise ValueError(f"Unsupported config extension: {path.suffix}")


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run DOE experiments")
    parser.add_argument("--exp", type=pathlib.Path, required=True)
    parser.add_argument("--base", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument(
        "--processes",
        action="store_true",
        help="Use a process pool instead of threads for parallel execution",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-evaluate configurations even if present in the run index",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    run(
        args.exp,
        args.base,
        args.out,
        args.parallel,
        args.processes,
        force=args.force,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
