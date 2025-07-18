import numpy as np
from typing import List, Optional, Dict


class NodeManager:
    """Manage node attributes using NumPy arrays for fast vectorized operations.

    Arrays are pre-allocated and resized dynamically as nodes are added. The
    ``size`` attribute tracks how many slots are currently in use.
    """

    def __init__(self, initial_capacity: int = 16) -> None:
        """Create a manager with an optional starting *initial_capacity*."""

        self.capacity = max(initial_capacity, 1)
        self.size = 0
        self.node_ids: List[str] = []
        self.positions = np.zeros((self.capacity, 2), dtype=float)
        self.frequency = np.zeros(self.capacity, dtype=float)
        self.phase = np.zeros(self.capacity, dtype=float)
        self.coherence = np.ones(self.capacity, dtype=float)
        self.decoherence = np.zeros(self.capacity, dtype=float)
        self._index: Dict[str, int] = {}

    def _ensure_capacity(self) -> None:
        if self.size >= self.capacity:
            new_capacity = self.capacity * 2
            self.positions = np.vstack(
                [self.positions, np.zeros((new_capacity - self.capacity, 2))]
            )
            self.frequency = np.concatenate(
                [self.frequency, np.zeros(new_capacity - self.capacity)]
            )
            self.phase = np.concatenate(
                [self.phase, np.zeros(new_capacity - self.capacity)]
            )
            self.coherence = np.concatenate(
                [self.coherence, np.ones(new_capacity - self.capacity)]
            )
            self.decoherence = np.concatenate(
                [self.decoherence, np.zeros(new_capacity - self.capacity)]
            )
            self.capacity = new_capacity

    def add_node(
        self,
        node_id: str,
        x: float = 0.0,
        y: float = 0.0,
        frequency: float = 1.0,
        phase: float = 0.0,
    ) -> None:
        """Add a new node to the manager."""
        if node_id in self._index:
            raise ValueError(f"Node '{node_id}' already exists")
        self._ensure_capacity()
        self.node_ids.append(node_id)
        self._index[node_id] = self.size
        self.positions[self.size] = [x, y]
        self.frequency[self.size] = frequency
        self.phase[self.size] = phase
        self.coherence[self.size] = 1.0
        self.decoherence[self.size] = 0.0
        self.size += 1

    def remove_node(self, node_id: str) -> None:
        """Remove a node and its data."""
        idx = self._index.pop(node_id, None)
        if idx is None:
            return
        last_idx = self.size - 1
        last_id = self.node_ids[last_idx]
        if idx != last_idx:
            self.positions[idx] = self.positions[last_idx]
            self.frequency[idx] = self.frequency[last_idx]
            self.phase[idx] = self.phase[last_idx]
            self.coherence[idx] = self.coherence[last_idx]
            self.decoherence[idx] = self.decoherence[last_idx]
            self.node_ids[idx] = last_id
            self._index[last_id] = idx
        self.node_ids.pop()
        self.size -= 1

    def increment_coherence(self, amount: float) -> None:
        """Increase coherence of all nodes by ``amount`` using vectorization."""
        self.coherence += amount

    def update_phase(self, delta: float) -> None:
        """Increment all node phases by ``delta``."""
        self.phase += delta

    def get_node_index(self, node_id: str) -> Optional[int]:
        """Return the array index for ``node_id`` or ``None`` if missing."""
        return self._index.get(node_id)
