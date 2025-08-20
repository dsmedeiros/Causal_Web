"""Ensure golden run logs replay deterministically."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from Causal_Web.engine.engine_v2.adapter import EngineAdapter

RUN_DIR = Path("tests/golden_replays")
RUNS = sorted(p for p in RUN_DIR.iterdir() if p.is_dir())


@pytest.mark.parametrize("run_dir", RUNS)
def test_golden_replay_matches_logs(run_dir: Path) -> None:
    """Replay logged deltas and verify frames and residuals match."""
    expected_frames: list[int] = []
    expected_residuals: list[float] = []
    with (run_dir / "delta_log.jsonl").open() as fh:
        for line in fh:
            d = json.loads(line)
            expected_frames.append(d.get("frame", 0))
            expected_residuals.append(d["counters"]["residual"])

    adapter = EngineAdapter()
    adapter.load_replay(run_dir)
    adapter._replay_playing = True

    frames: list[int] = []
    residuals: list[float] = []
    while True:
        d = adapter.snapshot_delta()
        if d is None:
            break
        frames.append(d.get("frame", 0))
        residuals.append(d["counters"]["residual"])

    assert frames == expected_frames
    assert residuals == expected_residuals
