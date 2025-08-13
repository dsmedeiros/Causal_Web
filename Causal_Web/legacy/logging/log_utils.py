"""No-op utilities for legacy logging hooks."""

from __future__ import annotations
import warnings
from typing import Any

warnings.warn(
    "Causal_Web.legacy.logging.log_utils is deprecated and will be removed",
    DeprecationWarning,
    stacklevel=2,
)


def attach_graph(graph: Any) -> None:
    """Placeholder that ignores the provided graph."""
    return None


def log_meta_node_ticks(tick: int) -> None:
    """No-op for legacy tick logging."""
    return None


def log_curvature_per_tick(tick: int) -> None:
    """No-op for legacy curvature logging."""
    return None


def snapshot_graph(tick: int) -> dict[str, Any]:
    """Return an empty snapshot for compatibility."""
    return {}


def write_output() -> None:
    """No-op retained for compatibility."""
    return None


def log_bridge_states(global_tick: int) -> None:
    """No-op for legacy bridge state logging."""
    return None


def log_metrics_per_tick(tick: int, metrics: dict[str, float] | None = None) -> None:
    """No-op for per-tick metric logging."""
    return None


def export_curvature_map(graph: Any) -> None:
    """Placeholder that does nothing."""
    return None


def export_global_diagnostics(graph: Any) -> None:
    """Placeholder that does nothing."""
    return None


def export_regional_maps(graph: Any) -> None:
    """Placeholder that does nothing."""
    return None
