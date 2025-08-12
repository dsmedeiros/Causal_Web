"""No-op utilities for legacy logging hooks."""

from __future__ import annotations
from typing import Any


def attach_graph(graph: Any) -> None:
    return None


def log_meta_node_ticks(tick: int) -> None:
    return None


def log_curvature_per_tick(tick: int) -> None:
    return None


def snapshot_graph(tick: int) -> dict[str, Any]:
    return {}


def write_output() -> None:
    return None


def log_bridge_states(global_tick: int) -> None:
    return None


def log_metrics_per_tick(tick: int, metrics: dict[str, float] | None = None) -> None:
    return None


def export_curvature_map(graph: Any) -> None:
    return None


def export_global_diagnostics(graph: Any) -> None:
    return None


def export_regional_maps(graph: Any) -> None:
    return None
