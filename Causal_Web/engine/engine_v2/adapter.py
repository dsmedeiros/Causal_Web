"""Compatibility layer for the v2 engine prototype.

The :class:`EngineAdapter` exposes a subset of the legacy tick engine API but
drives a new depth-based scheduler and the lightweight :mod:`lccm` model.  The
adapter processes packets ordered by arrival depth and advances vertex windows
according to the local causal consistency math.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from .lccm import LCCM
from .scheduler import DepthScheduler
from .state import Packet, TelemetryFrame


class EngineAdapter:
    """Bridge between the legacy orchestrator calls and engine v2."""

    def __init__(self) -> None:
        self._scheduler = DepthScheduler()
        self._running = False
        self._vertices: Dict[int, Dict[str, Any]] = {}

    # Public API -----------------------------------------------------
    def build_graph(self, graph_json_path: str | Dict[str, Any]) -> None:
        """Load a graph description."""

        if isinstance(graph_json_path, dict):
            graph = graph_json_path
        else:  # pragma: no cover - simple file IO
            import json

            with open(graph_json_path, "r", encoding="utf-8") as fh:
                graph = json.load(fh)

        params = graph.get("params", {})
        W0 = params.get("W0", 4)
        zeta1 = params.get("zeta1", 0.0)
        zeta2 = params.get("zeta2", 0.0)
        rho0 = params.get("rho0", 1.0)
        a = params.get("a", 1.0)
        b = params.get("b", 0.5)
        C_min = params.get("C_min", 0.0)
        f_min = params.get("f_min", 1.0)
        H_max = params.get("H_max", 0.0)
        T_hold = params.get("T_hold", 1)
        T_class = params.get("T_class", 1)

        self._vertices.clear()
        for v in graph.get("vertices", []):
            vid = int(v["id"])
            edges = v.get("edges", [])
            deg = len(edges)
            rho_mean = v.get("rho_mean", 0.0)
            lccm = LCCM(
                W0,
                zeta1,
                zeta2,
                rho0,
                a,
                b,
                C_min,
                f_min,
                H_max,
                T_hold,
                T_class,
                deg=deg,
                rho_mean=rho_mean,
            )
            self._vertices[vid] = {"edges": edges, "lccm": lccm}

    def start(self) -> None:
        """Mark the engine as running."""

        self._running = True

    def pause(self) -> None:
        """Pause execution."""

        self._running = False

    def stop(self) -> None:
        """Stop execution and reset all state."""

        self._running = False
        self._vertices.clear()
        self._scheduler.clear()

    def step(self, max_events: int | None = None) -> TelemetryFrame:
        """Advance the simulation until a window rolls or ``max_events``."""

        return self.run_until_next_window_or(max_events)

    def run_until_next_window_or(self, limit: int | None) -> TelemetryFrame:
        """Run until the next window boundary or until ``limit`` events."""

        if not self._running:
            self.start()

        if limit is None:
            limit = float("inf")

        start_windows = {
            vid: data["lccm"].window_idx for vid, data in self._vertices.items()
        }
        events = 0
        packets = []

        while self._scheduler and events < limit:
            depth_arr, dst, edge_id, pkt = self._scheduler.pop()
            vertex = self._vertices.get(dst)
            if vertex is None:
                continue
            lccm = vertex["lccm"]
            lccm.advance_depth(depth_arr)
            lccm.deliver()
            packets.append(pkt)
            for edge in vertex.get("edges", []):
                depth_next = depth_arr + edge.get("d_eff", 1)
                new_pkt = Packet(src=dst, dst=edge["dst"], payload=None)
                self._scheduler.push(depth_next, edge["dst"], edge["id"], new_pkt)
            events += 1

            if any(
                data["lccm"].window_idx != start_windows[vid]
                for vid, data in self._vertices.items()
            ):
                break

        max_depth = 0
        for data in self._vertices.values():
            max_depth = max(max_depth, data["lccm"].depth)

        return TelemetryFrame(depth=max_depth, events=events, packets=packets)

    def snapshot_for_ui(self) -> dict:
        """Return a minimal snapshot for the GUI."""

        max_depth = 0
        for data in self._vertices.values():
            max_depth = max(max_depth, data["lccm"].depth)
        return {"depth": max_depth}

    def current_depth(self) -> int:
        """Return the current depth of the simulation."""

        max_depth = 0
        for data in self._vertices.values():
            max_depth = max(max_depth, data["lccm"].depth)
        return max_depth


__all__ = ["EngineAdapter"]
