"""Bell inequality analysis utilities."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Tuple, List

from ..config import Config


def compute_bell_statistics(
    log_file: str | None = None,
) -> Tuple[float, Dict[Tuple[str, str], float], Dict[str, Any]]:
    """Return CHSH ``S`` value from ``entangled_log.jsonl``.

    Parameters
    ----------
    log_file:
        Optional path to ``entangled_log.jsonl``. Defaults to the current
        run directory.

    Returns
    -------
    float
        The calculated ``S`` value.
    Dict[Tuple[str, str], float]
        Mapping of setting pair to expectation value ``E(A_i,B_j)``.
    Dict[str, Any]
        Optional metadata fields captured from the log.
    """

    log_path = Path(log_file or Config.output_path("entangled_log.jsonl"))
    if not log_path.exists():
        return 0.0, {}, {}

    events: List[dict] = []
    metadata: Dict[str, Any] = {}
    with log_path.open() as fh:
        for line in fh:
            if not line.strip():
                continue
            obj = json.loads(line)
            if (
                obj.get("event_type") == "measurement"
                or obj.get("label") == "measurement"
            ):
                payload = obj.get("payload") or obj.get("value") or {}
                if not metadata:
                    for key in [
                        "mode",
                        "kappa_a",
                        "kappa_xi",
                        "h_prefix_len",
                        "delta_ttl",
                        "batch_id",
                    ]:
                        if key in payload:
                            metadata[key] = payload[key]
                events.append(
                    {
                        "tick": obj.get("tick"),
                        "entangled_id": payload.get("entangled_id"),
                        "observer_id": payload.get("observer_id"),
                        "setting": payload.get("measurement_setting"),
                        "outcome": payload.get("binary_outcome"),
                    }
                )

    if not events:
        return 0.0, {}, metadata

    # Determine observers and setting labels
    observer_ids = sorted({e["observer_id"] for e in events})[:2]
    if len(observer_ids) < 2:
        return 0.0, {}, metadata
    obs_map = {observer_ids[0]: "A", observer_ids[1]: "B"}
    setting_names: Dict[str, Dict[float, str]] = defaultdict(dict)

    for e in events:
        obs = e["observer_id"]
        if obs not in obs_map:
            continue
        names = setting_names[obs]
        if e["setting"] not in names:
            label = "1" if not names else "2"
            names[e["setting"]] = f"{obs_map[obs]}{label}"

    # Group events by entangled_id and tick
    pairs: Dict[Tuple[str, int], Dict[str, Tuple[str, int]]] = defaultdict(dict)
    for e in events:
        obs = e["observer_id"]
        if obs not in obs_map:
            continue
        setting_label = setting_names[obs][e["setting"]]
        key = (e["entangled_id"], int(e["tick"]))
        pairs[key][obs_map[obs]] = (setting_label, int(e["outcome"]))

    # Count outcome combinations per setting pair
    counts: Dict[Tuple[str, str], Dict[Tuple[int, int], int]] = defaultdict(
        lambda: defaultdict(int)
    )
    for pair in pairs.values():
        if "A" not in pair or "B" not in pair:
            continue
        a_setting, a_out = pair["A"]
        b_setting, b_out = pair["B"]
        counts[(a_setting, b_setting)][(a_out, b_out)] += 1

    expectations: Dict[Tuple[str, str], float] = {}
    for key, combo in counts.items():
        n_pp = combo.get((1, 1), 0)
        n_pm = combo.get((1, -1), 0)
        n_mp = combo.get((-1, 1), 0)
        n_mm = combo.get((-1, -1), 0)
        total = n_pp + n_pm + n_mp + n_mm
        if total:
            expectations[key] = (n_pp - n_pm - n_mp + n_mm) / total
        else:
            expectations[key] = 0.0

    e_a1_b1 = expectations.get(("A1", "B1"), 0.0)
    e_a1_b2 = expectations.get(("A1", "B2"), 0.0)
    e_a2_b1 = expectations.get(("A2", "B1"), 0.0)
    e_a2_b2 = expectations.get(("A2", "B2"), 0.0)
    s_value = e_a1_b1 - e_a1_b2 + e_a2_b1 + e_a2_b2

    return s_value, expectations, metadata
