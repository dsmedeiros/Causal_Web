"""Lightweight data structures for the experimental engine.

The module defines Struct-of-Arrays containers used by the
prototype scheduler along with simple packet and telemetry frame
representations. The structures are intentionally minimal and
will expand as the engine matures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class VertexArray:
    """Struct-of-arrays representation of vertex data."""

    ids: list[int] = field(default_factory=list)


@dataclass
class EdgeArray:
    """Struct-of-arrays representation of edge data."""

    ids: list[int] = field(default_factory=list)


@dataclass
class Packet:
    """A scheduled message between two vertices."""

    src: int
    dst: int
    payload: Any | None = None


@dataclass
class TelemetryFrame:
    """Snapshot returned to the GUI after each step."""

    depth: int
    events: int
    packets: list[Packet] = field(default_factory=list)
    window: int = 0
