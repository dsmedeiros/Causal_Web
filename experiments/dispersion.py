"""Dispersion relation utilities for split-step walks.

The functions here provide a light-weight numerical routine for
estimating the dispersion relation of a split-step quantum walk.  A
closed form for the 1D case is used which makes the routine fast and
deterministic.
"""

from __future__ import annotations

import csv
from typing import Dict, Iterable, List

import numpy as np

from Causal_Web.config import Config


def compute_dispersion(
    k_values: Iterable[float], theta1: float, theta2: float
) -> List[Dict[str, float]]:
    """Return ``ω(k)`` and the group velocity for each entry in ``k_values``.

    Parameters
    ----------
    k_values:
        Iterable of wave numbers.
    theta1, theta2:
        Split-step coin rotation angles.
    """

    k = np.asarray(list(k_values), dtype=float)
    cos_omega = np.cos(theta1) * np.cos(theta2) * np.cos(k) + np.sin(theta1) * np.sin(
        theta2
    )
    omega = np.arccos(np.clip(cos_omega, -1.0, 1.0))
    vg = np.gradient(omega, k) if len(k) > 1 else np.zeros_like(k)
    return [
        {"k": float(k[i]), "omega": float(omega[i]), "group_velocity": float(vg[i])}
        for i in range(len(k))
    ]


def run_dispersion(output_path: str) -> List[Dict[str, float]]:
    """Execute a dispersion sweep and persist the results to ``output_path``.

    The function reads parameters from :class:`~Causal_Web.config.Config`
    and writes a CSV file containing the measured points together with a
    mode tag and parameters for telemetry.
    """

    if not Config.qwalk.get("enabled", False):
        raise RuntimeError("qwalk is disabled")
    theta1 = float(Config.qwalk["thetas"]["theta1"])
    theta2 = float(Config.qwalk["thetas"]["theta2"])
    k_vals = Config.dispersion.get("k_values", [0.0])
    rows = compute_dispersion(k_vals, theta1, theta2)
    for r in rows:
        r.update({"mode": "dispersion", "theta1": theta1, "theta2": theta2})
    with open(output_path, "w", newline="") as fh:
        writer = csv.DictWriter(
            fh, ["mode", "theta1", "theta2", "k", "omega", "group_velocity"]
        )
        writer.writeheader()
        writer.writerows(rows)
    return rows
