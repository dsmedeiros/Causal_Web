"""Causal_Web package initialization."""

from __future__ import annotations

from typing import Any

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - imported for typing only
    from .engine.engine_v2.adapter import EngineAdapter

__all__ = ["EngineAdapter"]


def __getattr__(name: str) -> Any:  # pragma: no cover - attribute access
    """Lazily expose EngineAdapter and guard removed symbols."""

    if name == "EngineAdapter":
        from .engine.engine_v2.adapter import EngineAdapter as _EngineAdapter

        return _EngineAdapter
    if name == "NodeManager":
        raise RuntimeError("NodeManager has been removed; use EngineAdapter instead")
    raise AttributeError(name)
