import json
from collections import defaultdict
from pathlib import Path

import pytest

from Causal_Web.analysis.bell import compute_bell_statistics


def _write_log(
    path: Path, expectations: dict[tuple[float, float], float], N: int = 100
) -> None:
    tick = 0
    with path.open("w") as fh:
        for (a_setting, b_setting), E in expectations.items():
            n_pp = int(N * (1 + E) / 4)
            n_pm = int(N * (1 - E) / 4)
            n_mp = n_pm
            n_mm = n_pp
            for _ in range(n_pp):
                fh.write(
                    json.dumps(
                        {
                            "tick": tick,
                            "event_type": "measurement",
                            "value": {
                                "observer_id": "A",
                                "entangled_id": "E",
                                "measurement_setting": a_setting,
                                "binary_outcome": 1,
                            },
                        }
                    )
                    + "\n"
                )
                fh.write(
                    json.dumps(
                        {
                            "tick": tick,
                            "event_type": "measurement",
                            "value": {
                                "observer_id": "B",
                                "entangled_id": "E",
                                "measurement_setting": b_setting,
                                "binary_outcome": 1,
                            },
                        }
                    )
                    + "\n"
                )
                tick += 1
            for _ in range(n_pm):
                fh.write(
                    json.dumps(
                        {
                            "tick": tick,
                            "event_type": "measurement",
                            "value": {
                                "observer_id": "A",
                                "entangled_id": "E",
                                "measurement_setting": a_setting,
                                "binary_outcome": 1,
                            },
                        }
                    )
                    + "\n"
                )
                fh.write(
                    json.dumps(
                        {
                            "tick": tick,
                            "event_type": "measurement",
                            "value": {
                                "observer_id": "B",
                                "entangled_id": "E",
                                "measurement_setting": b_setting,
                                "binary_outcome": -1,
                            },
                        }
                    )
                    + "\n"
                )
                tick += 1
            for _ in range(n_mp):
                fh.write(
                    json.dumps(
                        {
                            "tick": tick,
                            "event_type": "measurement",
                            "value": {
                                "observer_id": "A",
                                "entangled_id": "E",
                                "measurement_setting": a_setting,
                                "binary_outcome": -1,
                            },
                        }
                    )
                    + "\n"
                )
                fh.write(
                    json.dumps(
                        {
                            "tick": tick,
                            "event_type": "measurement",
                            "value": {
                                "observer_id": "B",
                                "entangled_id": "E",
                                "measurement_setting": b_setting,
                                "binary_outcome": 1,
                            },
                        }
                    )
                    + "\n"
                )
                tick += 1
            for _ in range(n_mm):
                fh.write(
                    json.dumps(
                        {
                            "tick": tick,
                            "event_type": "measurement",
                            "value": {
                                "observer_id": "A",
                                "entangled_id": "E",
                                "measurement_setting": a_setting,
                                "binary_outcome": -1,
                            },
                        }
                    )
                    + "\n"
                )
                fh.write(
                    json.dumps(
                        {
                            "tick": tick,
                            "event_type": "measurement",
                            "value": {
                                "observer_id": "B",
                                "entangled_id": "E",
                                "measurement_setting": b_setting,
                                "binary_outcome": -1,
                            },
                        }
                    )
                    + "\n"
                )
                tick += 1


def _parse_counts(path: Path) -> dict[tuple[float, float], dict[tuple[int, int], int]]:
    pairs: dict[int, dict[str, tuple[float, int]]] = defaultdict(dict)
    with path.open() as fh:
        for line in fh:
            if not line.strip():
                continue
            obj = json.loads(line)
            payload = obj.get("value") or obj.get("payload")
            tick = int(obj["tick"])
            obs = payload["observer_id"]
            pairs[tick][obs] = (
                float(payload["measurement_setting"]),
                int(payload["binary_outcome"]),
            )
    counts: dict[tuple[float, float], dict[tuple[int, int], int]] = defaultdict(
        lambda: defaultdict(int)
    )
    for pair in pairs.values():
        if "A" in pair and "B" in pair:
            a_setting, a_out = pair["A"]
            b_setting, b_out = pair["B"]
            counts[(a_setting, b_setting)][(a_out, b_out)] += 1
    return counts


def _assert_no_signaling(
    counts: dict[tuple[float, float], dict[tuple[int, int], int]],
) -> None:
    for a_setting in {0.0, 1.0}:
        probs = []
        for b_setting in {0.0, 1.0}:
            combo = counts[(a_setting, b_setting)]
            total = sum(combo.values()) or 1
            p1 = (combo.get((1, 1), 0) + combo.get((1, -1), 0)) / total
            probs.append(p1)
        assert probs[0] == pytest.approx(probs[1], abs=1e-6)
    for b_setting in {0.0, 1.0}:
        probs = []
        for a_setting in {0.0, 1.0}:
            combo = counts[(a_setting, b_setting)]
            total = sum(combo.values()) or 1
            p1 = (combo.get((1, 1), 0) + combo.get((-1, 1), 0)) / total
            probs.append(p1)
        assert probs[0] == pytest.approx(probs[1], abs=1e-6)


def test_bell_mi_toggle(tmp_path: Path) -> None:
    strict_path = tmp_path / "strict.jsonl"
    conditioned_path = tmp_path / "conditioned.jsonl"

    strict_E = {
        (0.0, 0.0): 0.5,
        (0.0, 1.0): 0.5,
        (1.0, 0.0): 0.5,
        (1.0, 1.0): 0.5,
    }
    _write_log(strict_path, strict_E)
    s_val, _, _ = compute_bell_statistics(strict_path)
    assert s_val <= 2.0
    counts = _parse_counts(strict_path)
    _assert_no_signaling(counts)

    conditioned_E = {
        (0.0, 0.0): 0.7071,
        (0.0, 1.0): -0.7071,
        (1.0, 0.0): 0.7071,
        (1.0, 1.0): 0.7071,
    }
    _write_log(conditioned_path, conditioned_E)
    s_val, _, _ = compute_bell_statistics(conditioned_path)
    assert s_val > 2.0
    counts = _parse_counts(conditioned_path)
    _assert_no_signaling(counts)
