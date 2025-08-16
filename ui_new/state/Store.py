"""Store for static graph and latest snapshot delta."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class Store:
    """Maintain static graph data and the latest delta."""

    graph_static: Dict[str, Any] = field(default_factory=dict)
    latest_delta: Dict[str, Any] = field(default_factory=dict)

    def set_static(self, data: Dict[str, Any]) -> None:
        """Set immutable static graph information."""
        self.graph_static = data
        self.latest_delta.clear()

    def apply_delta(self, delta: Dict[str, Any]) -> None:
        """Coalesce ``delta`` into ``latest_delta`` so only the newest values remain."""
        for key, value in delta.items():
            if (
                key in self.latest_delta
                and isinstance(self.latest_delta[key], dict)
                and isinstance(value, dict)
            ):
                self.latest_delta[key].update(value)
            else:
                self.latest_delta[key] = value

    def current_state(self) -> Dict[str, Any]:
        """Return a merged view of static graph and latest delta."""
        state = self.graph_static.copy()
        state.update(self.latest_delta)
        return state
