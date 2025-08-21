"""Regression test for engine replay against golden logs."""

from __future__ import annotations

import os

import pytest

from Causal_Web.engine.replay import build_engine, replay_from_log
from tests.golden_utils import expected_from_log


def test_golden_replay(tmp_path):
    """Verify golden log replay reaches expected residual and frame count."""

    path = os.environ.get("GOLDEN_PATH", "tests/goldens/runA.jsonl")
    expected_frame, expected_residual = expected_from_log(path)
    engine = build_engine()
    frames = replay_from_log(engine, path, workdir=tmp_path)
    assert frames[-1].invariants["residual_ewma"] == pytest.approx(
        expected_residual, rel=0.15
    )
    assert frames[-1].frame == expected_frame
