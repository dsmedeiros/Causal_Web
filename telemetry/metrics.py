"""Metrics logging utilities."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import subprocess
from typing import Dict, Iterable, List

import numpy as np


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


_GIT = _git_commit()


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
            "git": _GIT,
            "ts": datetime.utcnow().isoformat() + "Z",
            **groups,
            **{f"raw_{k}": v for k, v in raw.items()},
            **gate_metrics,
            **invariants,
        }
        self.records.append(entry)

    def flush(self, cfg, samples: Iterable[Dict[str, float]]) -> None:
        """Write all stored metrics and summaries to disk."""

        self.out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = self.out_dir / "metrics.csv"
        if self.records:
            fieldnames: List[str] = list(self.records[0].keys())
            for rec in self.records[1:]:
                for key in rec.keys():
                    if key not in fieldnames:
                        fieldnames.append(key)
            with csv_path.open("w", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.records)

        def _aggregate(rows: List[Dict[str, object]]) -> Dict[str, float]:
            keys = {k for row in rows for k in row.keys() if k.startswith("G")}
            agg: Dict[str, float] = {}
            for k in sorted(keys):
                vals = [r[k] for r in rows if k in r]
                agg[f"mean_{k}"] = float(np.mean(vals))
                agg[f"std_{k}"] = float(np.std(vals))
            return agg

        summary = {
            "samples": cfg.samples,
            "groups": cfg.groups,
            "seed": cfg.seed,
            "gates": cfg.gates,
            "metrics_agg": _aggregate(self.records) if self.records else {},
        }
        (self.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

        inv_keys = {
            k for row in self.records for k in row.keys() if k.startswith("inv_")
        }
        if inv_keys:
            inv_summary: Dict[str, float] = {}
            for k in sorted(inv_keys):
                vals = [bool(r.get(k)) for r in self.records]
                inv_summary[k] = float(np.mean(vals))
            (self.out_dir / "summary_invariants.json").write_text(
                json.dumps(inv_summary, indent=2)
            )
