"""Ensure replay reproduces HUD metrics recorded during runs."""

from __future__ import annotations

import json
from pathlib import Path

from Causal_Web.engine.engine_v2.adapter import EngineAdapter
from Causal_Web.engine.engine_v2.state import Packet


def test_replay_matches_metrics(tmp_path: Path) -> None:
    """Record a short run and verify replay yields identical metrics."""

    # Run a minimal graph and log snapshot deltas
    adapter = EngineAdapter()
    graph = {
        "params": {"W0": 1},
        "nodes": [{"id": "0"}, {"id": "1"}],
        "edges": [{"from": "0", "to": "1", "delay": 1.0}],
    }
    adapter.build_graph(graph)
    adapter._scheduler.push(0, 0, 0, Packet(0, 1))

    deltas: list[dict] = []
    for _ in range(3):
        adapter.step(max_events=1)
        delta = adapter._build_delta()
        if delta:
            deltas.append(delta)

    # Persist run artifacts
    run_dir = tmp_path
    (run_dir / "graph_static.json").write_text(json.dumps({}))
    with (run_dir / "delta_log.jsonl").open("w") as fh:
        for d in deltas:
            fh.write(json.dumps(d) + "\n")

    frames = [d.get("frame", 0) for d in deltas]
    residuals = [d["counters"]["residual"] for d in deltas]

    # Load replay and stream logged deltas
    replay = EngineAdapter()
    replay.load_replay(run_dir)
    replay._replay_playing = True

    replay_frames: list[int] = []
    replay_residuals: list[float] = []
    while True:
        d = replay.snapshot_delta()
        if d is None:
            break
        replay_frames.append(d.get("frame", 0))
        replay_residuals.append(d["counters"]["residual"])

    assert replay_frames == frames
    assert replay_residuals == residuals
