import os
import threading
import time
from collections import defaultdict
from typing import Any, DefaultDict, List

from pydantic import BaseModel

from ...config import Config
from ..models.logging import GenericLogEntry


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
        if not Config.is_log_enabled(name):
            return
        logger.log(path, entry.model_dump_json() + "\n")


log_manager = LogManager()


def log_json(path: str, data: Any) -> None:
    """Buffer a JSON serialisable object as a newline delimited record."""
    name = os.path.basename(path)
    if not Config.is_log_enabled(name):
        return
    if isinstance(data, BaseModel):
        entry = data
    else:
        tick = data.get("tick", 0) if isinstance(data, dict) else 0
        entry = GenericLogEntry(
            event_type=name.replace(".json", ""), tick=tick, payload=data
        )
    log_manager.log(path, entry)
