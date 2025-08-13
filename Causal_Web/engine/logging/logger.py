from __future__ import annotations

"""Lightweight JSON line logger for the v2 engine."""

import json
import csv
from collections import Counter
from pathlib import Path
from typing import Any

from ...config import Config


class MetricAggregator:
    """Aggregate event counts per frame and write ``metrics.csv``."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.counts: Counter[str] = Counter()

    def add(self, frame: int, category: str) -> None:
        """Increment the count for ``category`` in ``frame``."""

        self.counts[category] += 1

    def flush(self, frame: int) -> None:
        """Write accumulated counts for ``frame`` to ``metrics.csv``."""

        self.path.parent.mkdir(parents=True, exist_ok=True)
        file_exists = self.path.exists()
        with self.path.open("a", newline="") as fh:
            fieldnames = ["frame", *sorted(self.counts.keys())]
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow({"frame": frame, **self.counts})
        self.counts.clear()


_AGGREGATOR: MetricAggregator | None = None


def _get_aggregator() -> MetricAggregator:
    global _AGGREGATOR
    if _AGGREGATOR is None:
        _AGGREGATOR = MetricAggregator(Path(Config.output_dir) / "metrics.csv")
    return _AGGREGATOR


def log_record(
    category: str,
    label: str,
    *,
    frame: int | None = None,
    tick: int | None = None,
    value: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    path: Path | None = None,
    **extra: Any,
) -> None:
    """Append a record to a JSON lines log file.

    ``frame`` is the preferred sequence identifier for new logs. ``tick`` is
    accepted for backward compatibility and copied verbatim when provided.
    """

    if path is None:
        path = Path(Config.output_dir) / f"{category}_log.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    data: dict[str, Any] = {"label": label}
    if frame is not None:
        data["frame"] = frame
    if tick is not None:
        data["tick"] = tick
    if value is not None:
        if isinstance(value, dict):
            data.update(value)
        else:
            data["value"] = value
    if metadata is not None:
        data["metadata"] = metadata
    if extra:
        data.update(extra)
    with path.open("a") as fh:
        fh.write(json.dumps(data) + "\n")
    if frame is not None and label != "adapter_frame":
        _get_aggregator().add(frame, category)


def log_json(
    category: str,
    label: str,
    payload: dict[str, Any],
    *,
    frame: int | None = None,
    tick: int | None = None,
) -> None:
    """Compatibility wrapper around :func:`log_record`.

    Parameters
    ----------
    category:
        Log category.
    label:
        Event label to write.
    payload:
        Data mapping to serialise.
    frame:
        Optional frame identifier included in ``payload``.
    tick:
        Optional legacy tick identifier included in ``payload``.
    """

    record = dict(payload)
    if frame is not None:
        record["frame"] = frame
    if tick is not None:
        record["tick"] = tick
    log_record(category, label, value=record)


class _LogManager:
    """Minimal stand-in for the legacy ``log_manager``/``logger``."""

    def flush(self) -> None:  # pragma: no cover - no state to flush
        return None


log_manager = _LogManager()
logger = log_manager


def flush_metrics(frame: int) -> None:
    """Flush aggregated metrics for ``frame`` to disk."""

    _get_aggregator().flush(frame)
