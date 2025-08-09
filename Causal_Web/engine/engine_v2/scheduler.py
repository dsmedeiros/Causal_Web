"""Event scheduler for the v2 engine prototype.

Events are ordered by a four-tuple key ``(depth_arr, dst_id, edge_id, seq)``
which allows deterministic processing of packets arriving at the same
depth. This module only provides a minimal priority queue wrapper
sufficient for early experimentation.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
from typing import Any, List, Tuple


@dataclass(order=True)
class _ScheduledItem:
    key: Tuple[int, int, int, int]
    payload: Any


class DepthScheduler:
    """Arrival-depth priority queue.

    Items are ordered by ``(depth_arr, dst_id, edge_id, seq)`` to guarantee a
    deterministic pop order even when multiple packets arrive at the same
    depth.  ``peek_depth`` exposes the depth of the next scheduled event
    without removing it from the queue, which is useful for detecting window
    boundaries in the adapter loop.
    """

    def __init__(self) -> None:
        self._queue: List[_ScheduledItem] = []
        self._seq = 0

    def push(self, depth_arr: int, dst_id: int, edge_id: int, payload: Any) -> None:
        """Insert a payload into the queue."""

        key = (depth_arr, dst_id, edge_id, self._seq)
        heapq.heappush(self._queue, _ScheduledItem(key, payload))
        self._seq += 1

    def pop(self) -> Tuple[int, int, int, Any]:
        """Remove and return the next scheduled payload and its metadata."""

        item = heapq.heappop(self._queue)
        depth_arr, dst_id, edge_id, _ = item.key
        return depth_arr, dst_id, edge_id, item.payload

    def peek_depth(self) -> int:
        """Return the arrival depth of the next payload without removing it."""

        return self._queue[0].key[0]

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._queue)

    def clear(self) -> None:
        """Drop all scheduled items."""

        self._queue.clear()
