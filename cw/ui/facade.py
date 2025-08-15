"""Facade providing safe engine access for UI clients."""

from __future__ import annotations

from typing import Any, Optional

from Causal_Web.engine.engine_v2.adapter import EngineAdapter


class EngineClient:
    """Thin wrapper around :class:`EngineAdapter` for UI consumers."""

    def __init__(self) -> None:
        self._adapter = EngineAdapter()

    # ------------------------------------------------------------------
    def build_graph(self, path: Optional[str] = None) -> None:
        if path is None:
            self._adapter.build_graph()
        else:
            self._adapter.build_graph(path)

    # ------------------------------------------------------------------
    def start(self) -> None:
        self._adapter.start()

    def step(self) -> None:
        self._adapter.step()

    def simulation_loop(self) -> None:
        self._adapter.simulation_loop()

    def pause(self) -> None:
        self._adapter.pause()

    def pause_simulation(self) -> None:
        self._adapter.pause_simulation()

    def resume_simulation(self) -> None:
        self._adapter.resume_simulation()

    def stop(self) -> None:
        self._adapter.stop()

    def stop_simulation(self) -> None:
        self._adapter.stop_simulation()

    # ------------------------------------------------------------------
    def current_frame(self) -> int:
        return self._adapter.current_frame()

    def snapshot_for_ui(self) -> Any:
        return self._adapter.snapshot_for_ui()

    @property
    def graph(self) -> Any:
        return self._adapter.graph

    def update_state(self, running: bool, paused: bool, tick: int, snap: Any) -> None:
        self._adapter._update_simulation_state(running, paused, tick, snap)
