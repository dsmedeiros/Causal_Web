"""Variational density update helpers."""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np

from ..rho_delay import effective_delay


def lambda_to_coeffs(
    lambda_s: float, lambda_l: float, lambda_I: float, eta: float
) -> tuple[float, float, float]:
    """Map variational weights to diffusion, leak and input coefficients."""
    total = lambda_s + lambda_l + lambda_I
    if total <= 0:
        raise ValueError("sum of lambda coefficients must be positive")
    alpha_d = lambda_s / total
    alpha_leak = lambda_l / total
    eta_eff = lambda_I * eta / total
    return alpha_d, alpha_leak, eta_eff


def coefficients(
    lambda_s: float, lambda_l: float, lambda_I: float, eta: float
) -> tuple[float, float, float]:
    """Return ``A``, ``B`` and ``C`` coefficients of the closed-form update."""
    total = lambda_s + lambda_l + lambda_I
    if total <= 0:
        raise ValueError("sum of lambda coefficients must be positive")
    A = lambda_s / total
    B = lambda_I / total
    C = lambda_I * eta / total
    return A, B, C


def update_rho_variational(
    rho: float,
    neighbours: Iterable[float],
    intensity: float,
    *,
    lambda_s: float,
    lambda_l: float,
    lambda_I: float,
    eta: float,
    d0: float,
    gamma: float,
    rho0: float,
) -> tuple[float, int]:
    """Update edge density using the variational objective."""
    if isinstance(neighbours, np.ndarray):
        mean = float(neighbours.mean()) if neighbours.size else 0.0
    else:
        nbr = list(neighbours)
        mean = sum(nbr) / len(nbr) if nbr else 0.0
    A, B, C = coefficients(lambda_s, lambda_l, lambda_I, eta)
    rho_new = A * mean + B * rho + C * intensity
    rho_new = max(0.0, rho_new)
    d_eff = effective_delay(rho_new, d0=d0, gamma=gamma, rho0=rho0)
    return rho_new, d_eff


def stamp_rho_metadata(
    meta: Dict[str, object], mode: str, params: Dict[str, object]
) -> None:
    """Annotate run metadata with the ρ update mode and parameters."""
    meta["rho_update"] = {"mode": mode, **params}


__all__ = [
    "lambda_to_coeffs",
    "coefficients",
    "update_rho_variational",
    "stamp_rho_metadata",
]
