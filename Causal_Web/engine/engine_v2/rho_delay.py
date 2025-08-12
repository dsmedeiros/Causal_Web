"""Density-dependent delay helpers.

Functions here provide helpers for evolving a stress--energy density
associated with edges and mapping that density to an effective delay.
The ``update_rho_delay`` routine implements a leaky diffusion step with
external input and a logarithmic delay scaling used by the experimental
engine.
"""

from __future__ import annotations

import math
from typing import Iterable, List, Tuple

import numpy as np


def diffuse(rho: List[float], weight: float) -> List[float]:
    """Diffuse density values across neighbours.

    Parameters
    ----------
    rho:
        List of density values.
    weight:
        Diffusion weight in ``[0, 1]``.
    """

    if not rho:
        return []
    avg = sum(rho) / len(rho)
    return [r + weight * (avg - r) for r in rho]


def effective_delay(
    rho: float,
    *,
    d0: float,
    gamma: float,
    rho0: float,
) -> float:
    """Map density to an effective delay using the saturating log rule."""

    return float(max(1, d0 + math.floor(gamma * math.log(1 + rho / rho0))))


def update_rho_delay(
    rho: float,
    neighbours: Iterable[float],
    intensity: float,
    *,
    alpha_d: float,
    alpha_leak: float,
    eta: float,
    d0: float,
    gamma: float,
    rho0: float,
) -> Tuple[float, int]:
    """Update density and effective delay for a single edge.

    Parameters
    ----------
    rho:
        Current density on the edge.
    neighbours:
        Densities of neighbouring edges used for diffusion. If provided as a
        NumPy array the mean is computed via vectorised operations to reduce
        Python overhead.
    intensity:
        External input contribution ``I``.
    alpha_d, alpha_leak, eta:
        Diffusion, leakage and input weights.
    d0, gamma, rho0:
        Baseline delay and scaling parameters for ``d_eff``.

    Returns
    -------
    tuple
        Updated ``rho`` and integer ``d_eff``.
    """

    if isinstance(neighbours, np.ndarray):
        mean = float(neighbours.mean()) if neighbours.size else 0.0
    else:
        nbr_list = list(neighbours)
        mean = sum(nbr_list) / len(nbr_list) if nbr_list else 0.0
    rho = (1 - alpha_d - alpha_leak) * rho + alpha_d * mean + eta * intensity
    rho = max(0.0, rho)
    d_eff = max(1, int(d0 + math.floor(gamma * math.log(1 + rho / rho0))))
    return rho, d_eff


def update_rho_delay_vec(
    rho: np.ndarray,
    mean: np.ndarray,
    intensity: np.ndarray | float,
    *,
    alpha_d: float,
    alpha_leak: float,
    eta: float,
    d0: np.ndarray,
    gamma: float,
    rho0: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised density and delay update.

    Parameters
    ----------
    rho:
        Current density values for each edge.
    mean:
        Mean neighbour density for each edge.
    intensity:
        External input for each edge. May be a scalar shared across updates or
        a vector matching the shape of ``rho`` for per-edge intensities.
    d0:
        Baseline delays per edge.

    Returns
    -------
    tuple of arrays
        Updated densities and integer effective delays.
    """

    rho = (1 - alpha_d - alpha_leak) * rho + alpha_d * mean + eta * intensity
    rho = np.maximum(0.0, rho)
    d_eff = np.maximum(1, np.floor(d0 + gamma * np.log(1 + rho / rho0))).astype(
        np.int32
    )
    return rho, d_eff


__all__ = ["diffuse", "effective_delay", "update_rho_delay", "update_rho_delay_vec"]
