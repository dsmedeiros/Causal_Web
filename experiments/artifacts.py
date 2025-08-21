from __future__ import annotations

"""Utility functions for experiment result artifacts.

This module persists and loads top-K and hall-of-fame records in a
simple JSON format.  The records help surface the best runs in the UI
and allow users to replay or promote configurations.
"""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple
from datetime import datetime
import json
import secrets
import yaml


@dataclass
class TopKEntry:
    """Description of a single run in the top-K list."""

    run_id: str
    fitness: float
    objectives: Dict[str, float]
    groups: Dict[str, float]
    toggles: Dict[str, str]
    seed: int
    path: str


# ---------------------------------------------------------------------------
def write_best_config(
    best_row: Dict[str, Any], path: str = "experiments/best_config.yaml"
) -> str:
    """Persist ``best_row`` details to a YAML config and return the path.

    Parameters
    ----------
    best_row:
        Mapping describing a run, typically from top-K summaries.
        Must include ``groups``, ``seed`` and ``run_id`` fields and may
        optionally include ``toggles``.
    path:
        Destination file path. Defaults to ``experiments/best_config.yaml``.

    Returns
    -------
    str
        The path that was written.
    """

    cfg = {
        "dimensionless": best_row["groups"],
        "toggles": best_row.get("toggles", {}),
        "seed": best_row["seed"],
        "notes": f"promoted from {best_row['run_id']}",
    }
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=True)
    return str(out_path)


# ---------------------------------------------------------------------------
def update_top_k(
    entries: Iterable[TopKEntry], path: Path, k: int = 50
) -> List[TopKEntry]:
    """Merge ``entries`` into existing top-K data and persist it.

    Parameters
    ----------
    entries:
        Iterable of new :class:`TopKEntry` objects.
    path:
        Destination JSON path.
    k:
        Maximum number of rows to retain.

    Returns
    -------
    List[TopKEntry]
        Sorted top-K list that was written to disk.
    """

    existing: List[TopKEntry] = []
    if path.exists():
        try:
            data = json.loads(path.read_text())
            for row in data.get("rows", []):
                existing.append(TopKEntry(**row))
        except Exception:
            pass
    combined = existing + list(entries)
    combined.sort(key=lambda e: e.fitness, reverse=True)
    top = combined[:k]
    payload = {"k": k, "rows": [asdict(e) for e in top]}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    return top


# ---------------------------------------------------------------------------
def load_top_k(path: Path) -> Dict[str, object]:
    """Load top-K data from ``path``."""

    if not path.exists():
        return {"k": 0, "rows": []}
    data = json.loads(path.read_text())
    return data


# ---------------------------------------------------------------------------
def save_hall_of_fame(entries: Iterable[Dict[str, object]], path: Path) -> None:
    """Persist ``entries`` to ``path`` in the hall-of-fame schema."""

    payload = {"archive": list(entries)}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


# ---------------------------------------------------------------------------
def load_hall_of_fame(path: Path) -> Dict[str, object]:
    """Load hall-of-fame data from ``path``."""

    if not path.exists():
        return {"archive": []}
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
def persist_run(
    config: Dict[str, Any],
    result: Dict[str, Any],
    path: Path,
    *,
    manifest: Dict[str, Any] | None = None,
    delta_log: Path | None = None,
) -> None:
    """Write ``config`` and ``result`` details to ``path``.

    A small directory is created containing ``config.json`` and ``result.json``
    files.  When ``manifest`` metadata is provided it is written to
    ``manifest.json`` to aid run indexing.  When available a ``delta_log.jsonl``
    capturing snapshot deltas is also copied so runs can be deterministically
    replayed.
    """

    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(json.dumps(config, indent=2))
    (path / "result.json").write_text(json.dumps(result, indent=2))
    if manifest is not None:
        (path / "manifest.json").write_text(json.dumps(manifest, indent=2))

    if delta_log is None:
        try:  # pragma: no cover - optional dependency
            from Causal_Web.config import Config

            delta_log = Path(Config.output_path("delta_log.jsonl"))
        except Exception:  # pragma: no cover - fallback when Config missing
            delta_log = None

    if delta_log and delta_log.exists():
        import shutil

        shutil.copy(delta_log, path / "delta_log.jsonl")


# ---------------------------------------------------------------------------
def allocate_run_dir(root: Path = Path("experiments")) -> Tuple[str, Path, str]:
    """Allocate a unique run directory under ``root``.

    Parameters
    ----------
    root:
        Root experiments directory. The run will be created inside
        ``root / runs / <date>``.

    Returns
    -------
    Tuple[str, Path, str]
        A tuple of ``(run_id, path, relative_path)`` where ``run_id`` is a
        timestamped identifier, ``path`` is the absolute directory path and
        ``relative_path`` is the path relative to ``root`` suitable for
        artifact records.
    """

    now = datetime.utcnow()
    token = secrets.token_hex(3)
    date = now.strftime("%Y-%m-%d")
    run_id = f"{now.strftime('%Y-%m-%dT%H-%M-%SZ')}-{token}"
    rel = Path("runs") / date / token
    abs_path = root / rel
    return run_id, abs_path, str(rel)


__all__ = [
    "TopKEntry",
    "update_top_k",
    "load_top_k",
    "save_hall_of_fame",
    "load_hall_of_fame",
    "write_best_config",
    "persist_run",
    "allocate_run_dir",
]
