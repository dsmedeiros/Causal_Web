"""Causal_Web package initialization."""

from __future__ import annotations

from typing import Any

from .engine.engine_v2.adapter import EngineAdapter

__all__ = ["EngineAdapter"]


def __getattr__(name: str) -> Any:  # pragma: no cover - attribute access
    """Provide informative error messages for removed legacy symbols."""

    if name == "NodeManager":
        raise RuntimeError("NodeManager has been removed; use EngineAdapter instead")
    raise AttributeError(name)


# TODO: legacy refactor
