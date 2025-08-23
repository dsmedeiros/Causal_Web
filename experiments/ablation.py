from __future__ import annotations

"""Utilities for local ablation around MCTS best configurations."""

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Sequence
import numpy as np


@dataclass
class AblationResult:
    """Store partial dependence results for one or two dimensions."""

    params: Sequence[str]
    values: Sequence[Sequence[float]]
    scores: Sequence[Sequence[float]]


def local_ablation(
    best: Dict[str, float],
    evaluate: Callable[[Dict[str, float]], float],
    features: Sequence[str],
    *,
    span: float = 0.1,
    steps: int = 20,
) -> AblationResult:
    """Compute partial dependence around ``best`` for ``features``.

    Parameters
    ----------
    best:
        Mapping of parameter names to their best discovered values.
    evaluate:
        Callable evaluating a configuration and returning a scalar score.
    features:
        One or two parameter names to ablate.
    span:
        Fractional range to sweep around the best value. A value of ``0.1``
        explores ``±10%`` around the centre.
    steps:
        Number of grid points per dimension.

    Returns
    -------
    AblationResult
        Structure containing sampled parameter values and corresponding
        scores.

    Notes
    -----
    The evaluation function is invoked ``steps`` times for one-dimensional
    ablations and ``steps**2`` times for two-dimensional slices.
    """

    if len(features) not in (1, 2):
        raise ValueError("features must contain one or two entries")

    centre = [best[f] for f in features]
    grids: List[np.ndarray] = []
    for c in centre:
        width = abs(c) * span
        if width == 0:
            width = span
        grids.append(np.linspace(c - width, c + width, steps))

    scores: List[List[float]] = []
    if len(features) == 1:
        param = features[0]
        vals = grids[0]
        for v in vals:
            cfg = dict(best)
            cfg[param] = float(v)
            scores.append([float(evaluate(cfg))])
        return AblationResult([param], [vals.tolist()], scores)

    # 2D ablation
    p1, p2 = features
    v1, v2 = grids
    for x in v1:
        row: List[float] = []
        for y in v2:
            cfg = dict(best)
            cfg[p1] = float(x)
            cfg[p2] = float(y)
            row.append(float(evaluate(cfg)))
        scores.append(row)
    return AblationResult([p1, p2], [v1.tolist(), v2.tolist()], scores)
