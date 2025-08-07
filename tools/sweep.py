from __future__ import annotations

"""Parameter sweep utilities producing CSV summaries and heat-maps."""

import argparse
import itertools
import logging
from pathlib import Path
from typing import Any, Dict, Iterable

import matplotlib.pyplot as plt
import pandas as pd
import yaml

plt.switch_backend("Agg")

from . import metrics


def _run_experiment(name: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Dispatch to the metric associated with ``name``."""

    if name == "bell":
        return {"bell": metrics.bell_score(**params)}
    if name == "interference":
        return {"visibility": metrics.interference_visibility(**params)}
    if name == "twin":
        res = metrics.tau_ratio(**params)
        return {"tau_ratio": res.ratio, "tau_expected": res.analytic}
    raise ValueError(f"unknown experiment {name}")


def _iter_grid(grid: Dict[str, Iterable[Any]]):
    keys = list(grid)
    for values in itertools.product(*(grid[k] for k in keys)):
        yield dict(zip(keys, values))


def _save_heatmap(df: pd.DataFrame, metric: str, out: Path) -> None:
    """Persist a heat-map for up to two parameter columns."""

    params = [c for c in df.columns if c not in {metric}]
    if len(params) > 2:
        logging.warning(
            "Heat-map supports only two parameters; ignoring %s", params[2:]
        )
    if len(params) == 1:
        p = params[0]
        plt.figure()
        plt.plot(df[p], df[metric], marker="o")
        plt.xlabel(p)
        plt.ylabel(metric)
        plt.savefig(out)
        plt.close()
    elif len(params) >= 2:
        x, y = params[:2]
        pivot = df.pivot(index=y, columns=x, values=metric)
        plt.figure()
        plt.imshow(pivot.values, origin="lower", aspect="auto")
        plt.xticks(range(len(pivot.columns)), pivot.columns)
        plt.yticks(range(len(pivot.index)), pivot.index)
        plt.xlabel(x)
        plt.ylabel(y)
        plt.colorbar(label=metric)
        plt.savefig(out)
        plt.close()


def sweep(config: Dict[str, Any]) -> None:
    """Execute sweeps defined in ``config``."""

    for name, spec in config.items():
        grid = {k: v for k, v in spec.items() if isinstance(v, list)}
        fixed = {k: v for k, v in spec.items() if not isinstance(v, list)}
        rows = []
        for combo in _iter_grid(grid):
            params = {**fixed, **combo}
            metrics_res = _run_experiment(name, params)
            rows.append({**combo, **metrics_res})
        df = pd.DataFrame(rows)
        csv_path = Path(f"{name}_sweep.csv")
        df.to_csv(csv_path, index=False)
        metric_col = next(iter(metrics_res))
        _save_heatmap(df, metric_col, Path(f"{name}_heatmap.png"))


def main(path: str) -> None:
    with open(path) as fh:
        config = yaml.safe_load(fh)
    sweep(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run parameter sweeps")
    parser.add_argument("config", type=str, help="YAML configuration file")
    args = parser.parse_args()
    main(args.config)
