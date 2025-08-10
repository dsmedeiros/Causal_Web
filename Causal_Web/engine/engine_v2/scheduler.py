"""Bucketed event scheduler for the v2 engine prototype."""

from __future__ import annotations

from bisect import bisect
from collections import defaultdict
import heapq
from typing import Any, DefaultDict, List, Tuple


class DepthScheduler:
    """Arrival-depth bucketed priority queue.

    Events are grouped into buckets by integer arrival depth.  Each bucket
    maintains its items ordered by ``(dst_id, edge_id, seq)`` ensuring a
    deterministic processing order. A separate min-heap tracks which depths are
    present, so heap operations occur only when a new depth is added or an
    existing depth bucket becomes empty, yielding amortised :math:`O(1)` push and
    pop operations for batches sharing the same depth.
    """

    def __init__(self) -> None:
        self._buckets: DefaultDict[int, List[Tuple[Tuple[int, int, int], Any]]] = (
            defaultdict(list)
        )
        self._depths: List[int] = []
        self._seq = 0

    def push(self, depth_arr: int, dst_id: int, edge_id: int, payload: Any) -> None:
        """Insert a payload into the scheduler."""

        bucket = self._buckets[depth_arr]
        if not bucket:
            heapq.heappush(self._depths, depth_arr)
        key = (dst_id, edge_id, self._seq)
        idx = bisect(bucket, (key, payload))
        bucket.insert(idx, (key, payload))
        self._seq += 1

    def pop(self) -> Tuple[int, int, int, Any]:
        """Remove and return the next scheduled payload and metadata."""

        if not self._depths:
            raise IndexError("pop from empty scheduler")
        depth = self._depths[0]
        bucket = self._buckets[depth]
        key, payload = bucket.pop(0)
        if not bucket:
            heapq.heappop(self._depths)
            del self._buckets[depth]
        dst_id, edge_id, _ = key
        return depth, dst_id, edge_id, payload

    def peek_depth(self) -> int:
        """Return the arrival depth of the next payload without removing it."""

        if not self._depths:
            raise IndexError("peek from empty scheduler")
        return self._depths[0]

    def __len__(self) -> int:  # pragma: no cover - trivial
        return sum(len(b) for b in self._buckets.values())

    def clear(self) -> None:
        """Drop all scheduled items."""

        self._buckets.clear()
        self._depths.clear()
        self._seq = 0
