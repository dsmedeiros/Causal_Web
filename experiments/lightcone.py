"""Lightcone envelope measurements for split-step walks."""

from __future__ import annotations

import csv
from typing import Dict, List

from Causal_Web.config import Config


def simulate_lightcone(max_distance: int, v_max: float = 1.0) -> List[Dict[str, float]]:
    """Return arrival depths for a pulse on a regular lattice.

    Parameters
    ----------
    max_distance:
        Largest graph distance to record.
    v_max:
        Assumed maximum propagation velocity.
    """

    return [
        {"mode": "lightcone", "distance": d, "arrival_depth": d / v_max}
        for d in range(max_distance + 1)
    ]


def run_lightcone(output_path: str) -> List[Dict[str, float]]:
    """Run the lightcone measurement and persist results to CSV."""

    max_d = int(Config.qwalk.get("max_distance", 10))
    rows = simulate_lightcone(max_d)
    with open(output_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, ["mode", "distance", "arrival_depth"])
        writer.writeheader()
        writer.writerows(rows)
    return rows
