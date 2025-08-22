"""Event-driven window sizing helpers for the LCCM.

This module implements a strictly local update rule for the LCCM window size
``W(v)``.  The rule maintains an exponentially weighted moving average (EWMA) of
the neighbour-density mean and maps it to a target window size that adapts to
both density and vertex degree.  The actual ``W(v)`` is rate-limited toward the
target to avoid oscillation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence
from numpy.typing import NDArray

import numpy as np


TRIM_MAX_FRACTION = 0.5


def robust_weighted_mean(
    values: Sequence[float] | NDArray[np.float64],
    weights: Sequence[float] | NDArray[np.float64],
    *,
    trim: float = 0.1,
) -> float:
    """Return a 10% trimmed weighted mean.

    The values are sorted and a fraction ``trim`` of the total weight is dropped
    from each tail before computing the weighted mean.  When ``trim`` is ``0``
    the regular weighted mean is returned.
    """

    if len(values) == 0:
        raise ValueError("values must be non-empty")
    if len(values) != len(weights):
        raise ValueError("values and weights must be the same length")
    if not 0 <= trim < TRIM_MAX_FRACTION:
        raise ValueError("trim fraction must be in [0, 0.5)")

    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)

    order = np.argsort(v)
    v = v[order]
    w = w[order]
    total = w.sum()
    if total <= 0:
        raise ValueError("weights must sum to a positive number")

    if trim > 0:
        cum = np.cumsum(w)
        lower = trim * total
        upper = (1.0 - trim) * total
        mask = (cum >= lower) & (cum <= upper)
        if not mask.any():
            # If everything was trimmed due to extreme weights just return simple mean
            return float(np.average(v, weights=w))
        v = v[mask]
        w = w[mask]
        # Adjust edge weights for partial trimming
        cum = cum[mask]
        w = w.copy()
        w[0] -= cum[0] - lower
        w[-1] -= upper - cum[-1]

    return float(np.average(v, weights=w))


@dataclass
class WindowParams:
    """Parameters controlling the adaptive window update."""

    W0: float = 8.0
    brho: float = 0.4
    rho0: float = 1.0
    bdeg: float = 0.2
    deg0: float = 3.0
    Wmin: float = 2.0
    Wmax: float = 64.0
    half_life_windows: float = 12.0
    beta: float | None = 0.1
    mu: float | None = None


@dataclass
class WindowState:
    """Per-vertex state for adaptive window sizing."""

    M_v: float = 0.0
    W_v: float = 8.0


def on_window_close(
    rhos: Sequence[float],
    weights: Sequence[float],
    params: WindowParams,
    state: WindowState,
    *,
    k: int = 1,
    deg_v: int | None = None,
) -> None:
    """Update the EWMA and window size when a vertex window closes.

    Parameters
    ----------
    rhos:
        Iterable of neighbour densities ``rho_u``.
    weights:
        Edge weights ``w_{v→u}`` corresponding to ``rhos``.  They do not need to
        be normalised; a normalisation step is performed internally.
    params:
        Static parameters controlling the update rule.
    state:
        Mutable state for the vertex.  ``state`` is updated in-place.
    k:
        Number of windows elapsed since the last update.  Defaults to ``1``.
    deg_v:
        Optional degree override.  If ``None`` the degree is inferred from the
        length of ``rhos``.
    """

    if len(rhos) == 0:
        return

    w = np.asarray(weights, dtype=float)
    if w.sum() <= 0:
        return
    w = w / w.sum()
    r = np.asarray(rhos, dtype=float)

    m_inst = robust_weighted_mean(r, w)

    alpha = math.log(2.0) / params.half_life_windows
    alpha_eff = 1.0 - (1.0 - alpha) ** k
    state.M_v = (1.0 - alpha_eff) * state.M_v + alpha_eff * m_inst

    deg = float(max(deg_v if deg_v is not None else len(r), 1))
    deg_term = 1.0 + params.bdeg * math.log1p(deg / params.deg0)
    rho_term = 1.0 + params.brho * math.log1p(state.M_v / params.rho0)

    W_target = params.W0 * rho_term * deg_term
    W_target = float(np.clip(W_target, params.Wmin, params.Wmax))

    if params.beta is not None:
        state.W_v = (1.0 - params.beta) * state.W_v + params.beta * W_target
    else:
        mu = params.mu if params.mu is not None else 0.1
        r_ratio = W_target / max(state.W_v, 1e-9)
        r_ratio = float(np.clip(r_ratio, 1.0 - mu, 1.0 + mu))
        state.W_v *= r_ratio


__all__ = [
    "WindowParams",
    "WindowState",
    "on_window_close",
    "robust_weighted_mean",
]
