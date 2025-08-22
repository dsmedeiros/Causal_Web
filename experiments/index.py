from __future__ import annotations

"""Run index and duplicate guard utilities.

This module exposes a stable :func:`run_key` used to identify experiment
runs across sweeps and genetic algorithm evaluations.  A small JSON index
tracks completed runs so subsequent submissions can skip configurations that
have already been evaluated.  The index is automatically rebuilt by scanning
existing ``runs/**/manifest.json`` files allowing safe resumption after
interruption.
"""

import hashlib
import json
import pathlib
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------


def run_key(cfg_dict: Dict[str, Any]) -> str:
    """Return a stable hash key for ``cfg_dict``.

    The dictionary should contain ``groups``, ``toggles``, ``seed`` and
    ``gates`` fields.  Keys are sorted and values JSON serialised using compact
    separators to ensure consistent hashing across processes.
    """

    s = json.dumps(cfg_dict, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(s.encode()).hexdigest()[:16]


class RunIndex:
    """Persisted mapping of ``run_key`` to run directories.

    Parameters
    ----------
    path:
        JSON file used to store the index.  Defaults to
        ``experiments/runs/index.json``.
    runs_root:
        Root directory containing run folders.  All ``manifest.json`` files
        under ``runs_root`` are scanned when rebuilding the index.
    """

    def __init__(
        self,
        path: pathlib.Path | None = None,
        runs_root: pathlib.Path | None = None,
    ) -> None:
        self.path = path or pathlib.Path("experiments/runs/index.json")
        self.runs_root = runs_root or self.path.parent
        self.seen: Dict[str, str] = {}
        if self.path.exists():
            try:
                self.seen = json.loads(self.path.read_text())
            except Exception:  # pragma: no cover - ignore corrupt index
                self.seen = {}
        self.rebuild()

    # ------------------------------------------------------------------
    def rebuild(self) -> None:
        """Rebuild the index by scanning run manifests."""

        self.seen = {}
        pattern = self.runs_root.glob("*/*/manifest.json")
        for manifest in pattern:
            try:
                data = json.loads(manifest.read_text())
                key = data.get("run_key")
                if key:
                    rel = str(manifest.parent.relative_to(self.runs_root))
                    self.seen[key] = rel
            except Exception:  # pragma: no cover - skip unreadable manifests
                continue
        self.save()

    # ------------------------------------------------------------------
    def save(self) -> None:
        """Persist the index to disk."""

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.seen, indent=2))

    # ------------------------------------------------------------------
    def mark(self, key: str, rel_path: str) -> None:
        """Record ``key`` with associated ``rel_path``."""

        rel = pathlib.Path(rel_path)
        try:
            rel = rel.relative_to(self.runs_root.name)
        except Exception:
            pass
        self.seen[key] = str(rel)
        self.save()

    # ------------------------------------------------------------------
    def get(self, key: str) -> Optional[str]:
        """Return run directory for ``key`` or ``None`` if unknown."""

        return self.seen.get(key)

    # ------------------------------------------------------------------
    def __contains__(self, key: str) -> bool:  # pragma: no cover - trivial
        return key in self.seen


__all__ = ["run_key", "RunIndex"]
