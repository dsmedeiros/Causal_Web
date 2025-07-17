import numpy as np
from typing import List, Optional, Dict


class NodeManager:
    """Manage node attributes using NumPy arrays for fast vectorized operations."""

    def __init__(self):
        self.node_ids: List[str] = []
        self.positions = np.zeros((0, 2), dtype=float)  # [[x, y], ...]
        self.frequency = np.array([], dtype=float)
        self.phase = np.array([], dtype=float)
        self.coherence = np.array([], dtype=float)
        self.decoherence = np.array([], dtype=float)
        self._index: Dict[str, int] = {}

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
        self.node_ids.append(node_id)
        self._index[node_id] = len(self.node_ids) - 1
        self.positions = np.vstack([self.positions, [x, y]])
        self.frequency = np.append(self.frequency, frequency)
        self.phase = np.append(self.phase, phase)
        self.coherence = np.append(self.coherence, 1.0)
        self.decoherence = np.append(self.decoherence, 0.0)

    def remove_node(self, node_id: str) -> None:
        """Remove a node and its data."""
        idx = self._index.pop(node_id, None)
        if idx is None:
            return
        self.node_ids.pop(idx)
        self.positions = np.delete(self.positions, idx, axis=0)
        self.frequency = np.delete(self.frequency, idx)
        self.phase = np.delete(self.phase, idx)
        self.coherence = np.delete(self.coherence, idx)
        self.decoherence = np.delete(self.decoherence, idx)
        # rebuild index
        self._index = {nid: i for i, nid in enumerate(self.node_ids)}

    def increment_coherence(self, amount: float) -> None:
        """Increase coherence of all nodes by ``amount`` using vectorization."""
        self.coherence += amount

    def update_phase(self, delta: float) -> None:
        """Increment all node phases by ``delta``."""
        self.phase += delta

    def get_node_index(self, node_id: str) -> Optional[int]:
        """Return the array index for ``node_id`` or ``None`` if missing."""
        return self._index.get(node_id)
