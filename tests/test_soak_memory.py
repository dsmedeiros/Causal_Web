"""Soak test ensuring memory usage remains bounded."""

from __future__ import annotations

import tracemalloc

from Causal_Web.engine.engine_v2.adapter import EngineAdapter
from Causal_Web.engine.engine_v2.state import Packet


def test_soak_memory_growth_under_threshold() -> None:
    """Run a long simulation and assert memory growth is small."""
    adapter = EngineAdapter()
    graph = {
        "params": {"W0": 1},
        "nodes": [{"id": "0"}, {"id": "1"}],
        "edges": [{"from": "0", "to": "1", "delay": 1.0}],
    }
    adapter.build_graph(graph)
    adapter._scheduler.push(0, 0, 0, Packet(0, 1))

    tracemalloc.start()
    start, _ = tracemalloc.get_traced_memory()
    for _ in range(50):
        adapter.step(max_events=1)
        adapter._build_delta()
    end, _ = tracemalloc.get_traced_memory()

    assert end - start < 1_000_000  # <1MB growth
