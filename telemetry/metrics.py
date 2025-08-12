"""Metrics logging utilities."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List


@dataclass
class MetricsLogger:
    """Collect metrics and emit them to disk."""

    out_dir: Path
    records: List[Dict[str, object]] = field(default_factory=list)

    def log(
        self, sample: int, groups: Dict[str, float], raw: Dict[str, float], seed: int
    ) -> None:
        """Store metrics for a single sample."""

        entry = {
            "sample": sample,
            "seed": seed,
            **groups,
            **{f"raw_{k}": v for k, v in raw.items()},
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
