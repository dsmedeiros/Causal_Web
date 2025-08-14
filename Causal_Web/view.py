"""UI-facing snapshot dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass(frozen=True)
class NodeView:
    """Lightweight representation of a node for the UI."""

    id: str


@dataclass(frozen=True)
class EdgeView:
    """Lightweight representation of an edge for the UI."""

    src: str
    dst: str


@dataclass(frozen=True)
class WindowEvent:
    """Event describing a closed window in the simulation."""

    v_id: str
    window_idx: int


@dataclass(frozen=True)
class ViewSnapshot:
    """Snapshot of engine state changes emitted to the GUI."""

    frame: int
    changed_nodes: List[NodeView] = field(default_factory=list)
    changed_edges: List[EdgeView] = field(default_factory=list)
    closed_windows: List[WindowEvent] = field(default_factory=list)
    counters: Dict[str, int] = field(default_factory=dict)
    invariants: Dict[str, float] = field(default_factory=dict)
