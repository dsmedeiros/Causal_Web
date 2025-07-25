from __future__ import annotations

"""Shared base classes and mixins for engine components."""

import os
from typing import Any

from ..config import Config
from .logger import log_json


class LoggingMixin:
    """Provide helper methods for JSON event logging."""

    def _log(self, name: str, record: dict[str, Any]) -> None:
        """Write ``record`` to the configured log ``name``."""
        log_json(Config.output_path(name), record)


class OutputDirMixin:
    """Provide an ``output_dir`` attribute and path resolution helper."""

    def __init__(self, output_dir: str | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        base = os.path.join(os.path.dirname(__file__), "..")
        self.output_dir = output_dir or os.path.join(base, "output")

    def _path(self, name: str) -> str:
        """Return absolute path for ``name`` inside :attr:`output_dir`."""
        return os.path.join(self.output_dir, name)
