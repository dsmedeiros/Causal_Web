from dataclasses import dataclass
from collections import deque
from typing import Deque

from ...config import Config


@dataclass
class Tick:
    """Discrete causal pulse with layer metadata."""

    origin: str
    time: float
    amplitude: float
    phase: float
    generation_tick: int = 0
    layer: str = "tick"
    trace_id: str = ""
    cumulative_delay: float = 0.0
    entangled_id: str | None = None


class TickPool:
    """Reusable pool of :class:`Tick` objects."""

    def __init__(self, size: int = 0) -> None:
        self._pool: Deque[Tick] = deque(Tick("", 0.0, 0.0, 0.0, 0) for _ in range(size))

    def acquire(self) -> Tick:
        """Return a tick instance from the pool or create a new one."""
        if self._pool:
            return self._pool.popleft()
        return Tick("", 0.0, 0.0, 0.0, 0)

    def release(self, tick: Tick) -> None:
        """Reset *tick* and put it back into the pool."""
        tick.origin = ""
        tick.time = 0.0
        tick.amplitude = 0.0
        tick.phase = 0.0
        tick.generation_tick = 0
        tick.layer = "tick"
        tick.trace_id = ""
        tick.cumulative_delay = 0.0
        tick.entangled_id = None
        self._pool.append(tick)


def create_global_pool() -> TickPool:
    """Instantiate the global tick pool using :data:`Config.TICK_POOL_SIZE`."""

    size = getattr(Config, "TICK_POOL_SIZE", 10000)
    return TickPool(size)


# Shared pool used across the simulation
GLOBAL_TICK_POOL = create_global_pool()
