"""Compare EHVI node values against scalarisation on Branin-Currin.

The benchmark runs MCTS-H with expected hypervolume improvement and with
Dirichlet scalarisation on the two-objective Branin-Currin function. The
final dominated hypervolumes are printed as JSON for a single deterministic
run using ``rng_seed=0``.
"""

from __future__ import annotations

import argparse
import json
from math import cos, exp, pi
from typing import Dict

import numpy as np

from experiments.optim import MCTS_H
from experiments.optim.priors import DiscretePrior


def branin(x: float, y: float) -> float:
    """Return the Branin function value for ``(x, y)`` in ``[0, 1]^2``."""
    x1 = x * 15 - 5
    x2 = y * 15
    a = 1.0
    b = 5.1 / (4 * pi**2)
    c = 5.0 / pi
    r = 6.0
    s = 10.0
    t = 1.0 / (8 * pi)
    return a * (x2 - b * x1**2 + c * x1 - r) ** 2 + s * (1 - t) * cos(x1) + s


def currin(x: float, y: float) -> float:
    """Return the Currin exponential function value for ``(x, y)``."""
    if y == 0:
        y = 1e-7
    return (1 - exp(-1 / (2 * y))) * (
        (2300 * x**3 + 1900 * x**2 + 2092 * x + 60)
        / (100 * x**3 + 500 * x**2 + 4 * x + 20)
    )


def objectives(x: float, y: float) -> Dict[str, float]:
    return {"branin": branin(x, y), "currin": currin(x, y)}


GRID = np.linspace(0.0, 1.0, 21).tolist()
PRIORS = {
    "x": DiscretePrior(GRID, [1 / len(GRID)] * len(GRID)),
    "y": DiscretePrior(GRID, [1 / len(GRID)] * len(GRID)),
}
HV_BOX = [[0, 310], [0, 15]]


def run(opt: MCTS_H, iters: int) -> float:
    for _ in range(iters):
        cfg = opt.suggest(1)[0]
        opt.observe([{"config": cfg, "objectives": objectives(cfg["x"], cfg["y"])}])
    return opt._hv


def hypervolume_scalarisation(iters: int) -> float:
    sc = MCTS_H(["x", "y"], PRIORS, {"multi_objective": True, "rng_seed": 0})
    points = []
    for _ in range(iters):
        cfg = sc.suggest(1)[0]
        objs = objectives(cfg["x"], cfg["y"])
        sc.observe([{"config": cfg, "objectives": objs}])
        points.append(np.array(list(objs.values())))
    sc.hv_box = np.array(HV_BOX, dtype=float)
    front: list[np.ndarray] = []
    for p in points:
        dominated = False
        for q in front:
            if np.all(q <= p) and np.any(q < p):
                dominated = True
                break
        if dominated:
            continue
        front = [q for q in front if not (np.all(p <= q) and np.any(p < q))]
        front.append(p)
    return sc._hv_value(front)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iters", type=int, default=40)
    args = parser.parse_args()

    hv = MCTS_H(
        ["x", "y"],
        PRIORS,
        {"multi_objective": True, "hv_box": HV_BOX, "rng_seed": 0},
    )
    hv_res = run(hv, args.iters)
    sc_res = hypervolume_scalarisation(args.iters)
    print(json.dumps({"ehvi": hv_res, "scalarisation": sc_res}))


if __name__ == "__main__":  # pragma: no cover
    main()
