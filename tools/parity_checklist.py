"""Phase 0 parity checklist for gate invariants.

This script runs the default set of gate benchmarks and prints the
resulting metrics. It helps verify that core invariants hold after
changes to the engine or UI.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Union

import sys


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import importlib.util


spec = importlib.util.spec_from_file_location(
    "experiments.gates", ROOT / "experiments" / "gates.py"
)
gates = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(gates)


def run_phase0() -> Dict[str, Union[float, bool]]:
    """Execute Phase 0 gates and return their metrics.

    Returns
    -------
    Dict[str, Union[float, bool]]
        Mapping of metric names to their numeric result or boolean
        invariant status.
    """

    cfg: Dict[str, Union[float, int]] = {
        "alpha_leak": 0.01,
        "eta": 0.01,
        "d0": 1.0,
        "gamma": 1.0,
        "rho0": 1.0,
    }
    return gates.run_gates(cfg, [1, 2, 3, 4, 5, 6])


if __name__ == "__main__":
    metrics = run_phase0()
    print(json.dumps(metrics, indent=2, sort_keys=True))
