from __future__ import annotations

"""Lightweight JSON line logger for the v2 engine."""

import json
from pathlib import Path
from typing import Any

from ...config import Config


def log_record(
    category: str,
    label: str,
    *,
    tick: int | None = None,
    value: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    path: Path | None = None,
    **extra: Any,
) -> None:
    """Append a record to a JSON lines log file.

    Parameters mirror those of the legacy logger but most fields are optional
    and recorded verbatim when provided.
    """

    if path is None:
        path = Path(Config.output_dir) / f"{category}_log.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    data: dict[str, Any] = {"label": label}
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


def log_json(
    category: str,
    label: str,
    payload: dict[str, Any],
    *,
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
    tick:
        Optional tick identifier included in ``payload``.
    """

    record = dict(payload)
    if tick is not None:
        record["tick"] = tick
    log_record(category, label, value=record)


class _LogManager:
    """Minimal stand-in for the legacy ``log_manager``/``logger``."""

    def flush(self) -> None:  # pragma: no cover - no state to flush
        return None


log_manager = _LogManager()
logger = log_manager
