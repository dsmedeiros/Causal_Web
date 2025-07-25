from __future__ import annotations

"""Shared base classes and mixins for engine components."""

from typing import Any

from ..config import Config
from .logger import log_json


class LoggingMixin:
    """Provide helper methods for JSON event logging."""

    def _log(self, name: str, record: dict[str, Any]) -> None:
        """Write ``record`` to the configured log ``name``."""
        log_json(Config.output_path(name), record)
