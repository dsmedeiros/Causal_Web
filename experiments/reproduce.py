"""Utilities for reproducing recorded experiment runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from invariants import checks
from .gates import run_gates


def reproduce_run(run_dir: Path) -> Dict[str, Any]:
    """Re-execute a run from ``run_dir`` and return the results.

    Parameters
    ----------
    run_dir:
        Path to a persisted run directory containing ``config.json`` and
        ``result.json`` files. ``manifest.json`` is optional but enables
        automatic discovery of gate selections and optimizer metadata.

    Returns
    -------
    Dict[str, Any]
        Mapping with reproduced ``metrics`` and ``invariants``. For MCTS runs
        a ``tree_hash`` key is populated when ``mcts_state.json`` is found.
    """

    config = json.loads((run_dir / "config.json").read_text())
    manifest_path = run_dir / "manifest.json"
    manifest: Dict[str, Any] = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    gates = manifest.get("gates", [])
    metrics = run_gates(config, gates)
    invariants = checks.from_metrics(metrics)

    tree_hash = None
    if manifest.get("optimizer") == "mcts_h":
        state_path = run_dir.parent.parent.parent / "mcts_state.json"
        if state_path.exists():
            state = json.loads(state_path.read_text())
            tree_hash = state.get("tree_hash")

    return {"metrics": metrics, "invariants": invariants, "tree_hash": tree_hash}
