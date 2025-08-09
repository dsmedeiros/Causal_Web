"""Compatibility layer for the v2 engine prototype.

The :class:`EngineAdapter` exposes a subset of the old tick engine API so
that existing entry points can drive the new core without modification.
All methods currently produce synthetic results; the real physics model
will fill in these hooks later.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from .scheduler import Scheduler
from .state import TelemetryFrame


class EngineAdapter:
    """Bridge between the legacy orchestrator calls and engine v2."""

    def __init__(self) -> None:
        self._scheduler = Scheduler()
        self._running = False
        self._depth = 0
        self._graph: Optional[Dict[str, Any]] = None

    # Public API -----------------------------------------------------
    def build_graph(self, graph_json_path: str | Dict[str, Any]) -> None:
        """Load a graph description.

        Parameters
        ----------
        graph_json_path:
            Path to a JSON graph file or an in-memory dictionary.
        """

        if isinstance(graph_json_path, dict):
            self._graph = graph_json_path
        else:  # pragma: no cover - simple file IO
            import json

            with open(graph_json_path, "r", encoding="utf-8") as fh:
                self._graph = json.load(fh)

    def start(self) -> None:
        """Mark the engine as running."""

        self._running = True

    def pause(self) -> None:
        """Pause execution."""

        self._running = False

    def stop(self) -> None:
        """Stop execution and reset all state."""

        self._running = False
        self._depth = 0
        self._scheduler.clear()

    def step(self, max_events: int | None = None) -> TelemetryFrame:
        """Advance the simulation by one synthetic step."""

        if not self._running:
            self.start()

        event_count = max_events or 0
        frame = TelemetryFrame(depth=self._depth, events=event_count)
        self._depth += 1
        self._scheduler.clear()
        return frame

    def run_until_next_window_or(self, limit: int | None) -> TelemetryFrame:
        """Run until the next window boundary or until ``limit`` events."""

        return self.step(max_events=limit)

    def snapshot_for_ui(self) -> dict:
        """Return a minimal snapshot for the GUI."""

        return {"depth": self._depth}

    def current_depth(self) -> int:
        """Return the current depth of the simulation."""

        return self._depth
