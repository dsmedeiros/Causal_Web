"""Contract tests for stream protocol MessagePack helpers."""

from __future__ import annotations

import msgpack
import pytest

from Causal_Web.engine.stream.protocol import (
    pack_graph_static,
    pack_snapshot_delta,
    unpack_graph_static,
    unpack_snapshot_delta,
)


def test_pack_unpack_graph_static() -> None:
    data = {"nodes": [{"id": "0"}]}
    raw = pack_graph_static(data)
    result = unpack_graph_static(raw)
    assert result == data


def test_unpack_graph_static_unknown_version() -> None:
    payload = {"type": "GraphStatic", "v": 99}
    raw = msgpack.packb(payload, use_bin_type=True, use_single_float=True)
    with pytest.raises(ValueError):
        unpack_graph_static(raw)


def test_unpack_graph_static_missing_v() -> None:
    payload = {"type": "GraphStatic"}
    raw = msgpack.packb(payload, use_bin_type=True, use_single_float=True)
    with pytest.raises(ValueError):
        unpack_graph_static(raw)


def test_pack_unpack_snapshot_delta() -> None:
    data = {"frame": 1, "counters": {"residual": 0.0}}
    raw = pack_snapshot_delta(data)
    result = unpack_snapshot_delta(raw)
    assert result == data


def test_unpack_snapshot_delta_unknown_version() -> None:
    payload = {"type": "SnapshotDelta", "v": 42}
    raw = msgpack.packb(payload, use_bin_type=True, use_single_float=True)
    with pytest.raises(ValueError):
        unpack_snapshot_delta(raw)


def test_unpack_snapshot_delta_missing_v() -> None:
    payload = {"type": "SnapshotDelta"}
    raw = msgpack.packb(payload, use_bin_type=True, use_single_float=True)
    with pytest.raises(ValueError):
        unpack_snapshot_delta(raw)
