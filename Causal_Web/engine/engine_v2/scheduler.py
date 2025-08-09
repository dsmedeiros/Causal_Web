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


class Scheduler:
    """Arrival-depth priority queue."""

    def __init__(self) -> None:
        self._queue: List[_ScheduledItem] = []
        self._seq = 0

    def push(self, depth_arr: int, dst_id: int, edge_id: int, payload: Any) -> None:
        """Insert a payload into the queue."""

        key = (depth_arr, dst_id, edge_id, self._seq)
        heapq.heappush(self._queue, _ScheduledItem(key, payload))
        self._seq += 1

    def pop(self) -> Any:
        """Remove and return the next payload."""

        return heapq.heappop(self._queue).payload

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._queue)

    def clear(self) -> None:
        """Drop all scheduled items."""

        self._queue.clear()
