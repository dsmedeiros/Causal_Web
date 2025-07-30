from __future__ import annotations

"""Shared base classes and mixins for engine components."""

import os
from typing import Any

from ...config import Config
from ..logging.logger import log_json
import json


class LoggingMixin:
    """Provide helper methods for JSON event logging."""

    def _log(self, name: str, record: dict[str, Any]) -> None:
        """Write ``record`` to the configured log ``name``."""
        category = Config.category_for_file(name)
        label = name.replace(".json", "")
        tick = record.get("tick")
        payload = {k: v for k, v in record.items() if k != "tick"}
        log_json(category, label, payload, tick=tick)


class PathLoggingMixin:
    """Provide helper method for logging to an explicit file path."""

    def _log_path(self, path: str, record: dict[str, Any]) -> None:
        """Write ``record`` directly to ``path``."""
        name = os.path.basename(path)
        category = Config.category_for_file(name)
        label = name.replace(".json", "")
        tick = record.get("tick")
        payload = {k: v for k, v in record.items() if k != "tick"}
        log_json(category, label, payload, tick=tick)


class OutputDirMixin:
    """Provide an ``output_dir`` attribute and path resolution helper."""

    def __init__(self, output_dir: str | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        base = os.path.join(os.path.dirname(__file__), "..")
        self.output_dir = output_dir or os.path.join(base, "output")

    def _path(self, name: str) -> str:
        """Return absolute path for ``name`` inside :attr:`output_dir`."""
        return os.path.join(self.output_dir, name)


class JsonLinesMixin:
    """Utility mixin to read newline-delimited JSON files."""

    @staticmethod
    def load_json_lines(path: str, int_keys: bool = False) -> dict:
        """Return dictionary loaded from newline-delimited JSON ``path``.

        Parameters
        ----------
        path:
            File containing one JSON object per line.
        int_keys:
            Convert digit-like keys to ``int`` when ``True``.
        """
        records: dict = {}
        if not os.path.exists(path):
            return records
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                # Case 1: single numeric key
                if len(obj) == 1 and next(iter(obj)).isdigit():
                    k = next(iter(obj))
                    records[int(k) if int_keys else k] = obj[k]
                    continue
                # Case 2: explicit tick field
                if "tick" in obj:
                    tick = int(obj.pop("tick"))
                    records[int(tick) if int_keys else str(tick)] = obj
                    continue
                # Fallback: numeric-like keys
                for k, v in obj.items():
                    if isinstance(k, str) and k.isdigit():
                        records[int(k) if int_keys else k] = v
        return records

    @staticmethod
    def load_event_log(path: str, int_keys: bool = False) -> dict:
        """Return list of events keyed by tick from ``path``."""
        records: dict = {}
        if not os.path.exists(path):
            return records
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                tick = int(obj.get("tick", 0))
                key = int(tick) if int_keys else str(tick)
                records.setdefault(key, []).append(obj)
        return records

    @staticmethod
    def filter_periodic_log(path: str, label: str, int_keys: bool = False) -> dict:
        """Return records with ``label`` from ``path`` keyed by tick."""

        result: dict = {}
        if not os.path.exists(path):
            return result
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if obj.get("label") != label:
                    continue
                tick = obj.get("tick")
                if tick is None:
                    continue
                key = int(tick) if int_keys else str(tick)
                result[key] = obj.get("value")
        return result

    @staticmethod
    def filter_event_log(path: str, event_type: str, int_keys: bool = False) -> dict:
        """Return events of ``event_type`` from ``path`` keyed by tick."""

        events = JsonLinesMixin.load_event_log(path, int_keys=int_keys)
        filtered: dict = {}
        for tick, lst in events.items():
            matches = [e for e in lst if e.get("event_type") == event_type]
            if matches:
                filtered[tick] = matches
        return filtered

    @staticmethod
    def group_periodic_log(path: str, int_keys: bool = False) -> dict:
        """Return all periodic records from ``path`` grouped by tick."""

        grouped: dict = {}
        if not os.path.exists(path):
            return grouped
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                tick = obj.get("tick")
                if tick is None:
                    continue
                key = int(tick) if int_keys else str(tick)
                grouped.setdefault(key, []).append(obj)
        return grouped
