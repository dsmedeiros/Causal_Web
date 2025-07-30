import os
import threading
import time
from collections import defaultdict
from typing import Any, DefaultDict, List

from pydantic import BaseModel

from ...config import Config
from ..models.logging import GenericLogEntry, PeriodicLogEntry


class LogBuffer:
    """Buffer log lines and flush them asynchronously."""

    def __init__(self, flush_interval: float = 1.0) -> None:
        self.flush_interval = flush_interval
        self._buffers: DefaultDict[str, List[str]] = defaultdict(list)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def log(self, path: str, line: str) -> None:
        """Append a raw line to the buffer for ``path``."""
        with self._lock:
            self._buffers[path].append(line)

    def flush(self) -> None:
        """Write all buffered lines to disk."""
        with self._lock:
            buffers = dict(self._buffers)
            self._buffers.clear()
        for path, lines in buffers.items():
            if not lines:
                continue
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "a") as f:
                f.writelines(lines)

    def _loop(self) -> None:
        while not self._stop.is_set():
            time.sleep(self.flush_interval)
            self.flush()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join()
        self.flush()


_interval = 0.1 if getattr(Config, "log_verbosity", "info") == "debug" else 1.0
logger = LogBuffer(flush_interval=_interval)


class LogManager:
    """Serialize and buffer validated log entries."""

    def log(self, path: str, entry: BaseModel) -> None:
        """Serialize ``entry`` and buffer it for ``path``."""
        name = os.path.basename(path)
        category = getattr(entry, "metadata", {}).get("category")
        label = name.replace(".jsonl", "").replace(".json", "")
        if not Config.is_log_enabled(category, label):
            return
        logger.log(path, entry.model_dump_json() + "\n")


log_manager = LogManager()


def log_record(
    category: str,
    label: str,
    tick: int | None = None,
    value: Any | None = None,
    mode: str | None = None,
    correlation_id: str | None = None,
    metadata: dict | None = None,
) -> None:
    """Serialize ``value`` as a log entry under ``category``.

    Parameters
    ----------
    category:
        Output category (``tick``, ``phenomena`` or ``event``).
    label:
        Logical name for the record.
    tick:
        Simulation tick associated with the record.
    value:
        Structured payload to persist.
    mode:
        Optional mode identifier stored with the record.
    correlation_id:
        Identifier linking this record to another event.
    metadata:
        Additional metadata to merge with default values.
    """

    meta = {"category": category}
    if mode is not None:
        meta["mode"] = mode
    if metadata:
        meta.update(metadata)

    if not Config.is_log_enabled(category, label):
        return

    path_map = {
        "tick": Config.output_path("ticks_log.jsonl"),
        "phenomena": Config.output_path("phenomena_log.jsonl"),
        "event": Config.output_path("events_log.jsonl"),
    }
    path = path_map.get(category, Config.output_path(f"{category}_log.jsonl"))

    if category in {"tick", "phenomena"}:
        entry = PeriodicLogEntry(tick=tick, label=label, value=value, metadata=meta)
    else:
        entry = GenericLogEntry(
            event_type=label,
            tick=tick or 0,
            payload=value,
            metadata=meta,
            correlation_id=correlation_id,
        )

    log_manager.log(path, entry)


def log_json(
    category: str,
    label: str,
    value: Any,
    tick: int | None = None,
    mode: str | None = None,
) -> None:
    """Backward compatible wrapper around :func:`log_record`."""
    log_record(
        category=category,
        label=label,
        tick=tick,
        value=value,
        mode=mode,
    )
