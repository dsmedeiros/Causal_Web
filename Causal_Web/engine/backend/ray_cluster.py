"""Minimal Ray helpers for sharding classical zones.

This module provides a tiny wrapper around :mod:`ray` to spread work across
multiple CPU or GPU workers.  Functions fall back to local execution when Ray
is not installed or a cluster is unavailable.
"""

from __future__ import annotations

from typing import Callable, Iterable, Any, List
import logging

try:  # pragma: no cover - optional dependency
    import ray
except Exception:  # pragma: no cover - gracefully handle missing Ray
    ray = None


def init_cluster(**kwargs: Any) -> None:
    """Initialise a Ray cluster if possible."""
    if ray is None:
        return
    if not ray.is_initialized():
        ray.init(**kwargs)


if ray is not None:  # pragma: no cover - decorator requires Ray

    @ray.remote
    def _apply(func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

else:  # pragma: no cover - local fallback

    def _apply(func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)


def map_zones(func: Callable[[Any], Any], zones: Iterable[Any]) -> List[Any]:
    """Execute ``func`` for each item in ``zones`` using Ray when available.

    Parameters
    ----------
    func:
        Callable applied to each zone.
    zones:
        Iterable of zone descriptors.

    Returns
    -------
    list
        Results for each zone.
    """
    zone_list = list(zones)
    if ray is None or not ray.is_initialized():
        logging.getLogger(__name__).info(
            "Ray cluster unavailable; running zones sequentially"
        )
        return [func(z) for z in zone_list]
    futures = [_apply.remote(func, z) for z in zone_list]
    return list(ray.get(futures))


def shutdown() -> None:
    """Shut down the Ray cluster if it is running."""
    if ray is not None and ray.is_initialized():
        ray.shutdown()
