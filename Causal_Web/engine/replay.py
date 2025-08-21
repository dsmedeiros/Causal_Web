"""Helpers for replaying engine runs in tests and tools."""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List

from .engine_v2.adapter import EngineAdapter


@dataclass
class Frame:
    """Simplified snapshot delta frame returned during replay."""

    frame: int
    invariants: dict[str, Any]


def build_engine() -> EngineAdapter:
    """Return a reset :class:`EngineAdapter` ready for replay.

    The adapter is initialised and issued a ``reset`` control command so each
    test starts from a clean slate without residual state from prior runs.
    """

    engine = EngineAdapter()
    engine.handle_control({"ExperimentControl": {"action": "reset"}})
    return engine


def replay_from_log(
    engine: EngineAdapter,
    path: str,
    *,
    max_frames: int | None = None,
    workdir: Path | None = None,
) -> List[Frame]:
    """Replay ``path`` through ``engine`` and return collected frames.

    Parameters
    ----------
    engine:
        Adapter used to drive the replay.
    path:
        Path to a line-delimited JSON delta log or a directory containing one.
    max_frames:
        Optional cap on the number of frames to collect.
    workdir:
        Directory used to stage replay files; a temporary directory is
        created if not supplied.
    """

    src = Path(path)
    temp_ctx: tempfile.TemporaryDirectory[str] | None = None
    if workdir is None:
        temp_ctx = tempfile.TemporaryDirectory()
        work = Path(temp_ctx.name)
    else:
        work = Path(workdir)
        work.mkdir(parents=True, exist_ok=True)

    if src.is_dir():
        shutil.copytree(src, work, dirs_exist_ok=True)
    else:
        shutil.copy(src, work / "delta_log.jsonl")

    engine.handle_control({"ExperimentControl": {"action": "reset"}})
    engine.handle_control({"ReplayControl": {"action": "load", "path": str(work)}})
    engine.handle_control({"ReplayControl": {"action": "play"}})
    frames: List[Frame] = []
    while True:
        delta = engine.snapshot_delta()
        if delta is None:
            break
        frames.append(
            Frame(frame=delta.get("frame", 0), invariants=delta.get("invariants", {}))
        )
        if max_frames is not None and len(frames) >= max_frames:
            break
    if temp_ctx is not None:
        temp_ctx.cleanup()
    return frames


__all__ = ["Frame", "build_engine", "replay_from_log"]
