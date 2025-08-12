"""Metrics logging utilities."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import subprocess
from typing import Dict, Iterable, List


@dataclass
class MetricsLogger:
    """Collect metrics and emit them to disk."""

    out_dir: Path
    records: List[Dict[str, object]] = field(default_factory=list)

    def log(
        self,
        sample: int,
        groups: Dict[str, float],
        raw: Dict[str, float],
        seed: int,
        gate_metrics: Dict[str, float | bool],
        invariants: Dict[str, float | bool],
    ) -> None:
        """Store metrics for a single sample."""

        entry = {
            "sample": sample,
            "seed": seed,
            "git": _git_commit(),
            "ts": datetime.utcnow().isoformat() + "Z",
            **groups,
            **{f"raw_{k}": v for k, v in raw.items()},
            **gate_metrics,
            **invariants,
        }
        self.records.append(entry)

    def flush(self, cfg, samples: Iterable[Dict[str, float]]) -> None:
        """Write all stored metrics to ``metrics.csv`` and ``summary.json``."""

        self.out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = self.out_dir / "metrics.csv"
        if self.records:
            with csv_path.open("w", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=self.records[0].keys())
                writer.writeheader()
                writer.writerows(self.records)
        summary = {
            "samples": cfg.samples,
            "groups": cfg.groups,
            "seed": cfg.seed,
        }
        (self.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))


def _git_commit() -> str:
    """Return the current git commit hash or ``"unknown"``."""

    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
            .decode()
            .strip()
        )
    except Exception:  # pragma: no cover - best effort only
        return "unknown"
