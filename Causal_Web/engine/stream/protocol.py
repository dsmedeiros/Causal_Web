"""MessagePack helpers for engine stream messages.

This module defines helpers for packing and unpacking ``GraphStatic`` and
``SnapshotDelta`` messages. Payloads are versioned via a ``v`` field and
include a ``type`` discriminator. Unknown versions or missing fields raise a
``ValueError`` to keep the IPC contract stable.
"""

from __future__ import annotations

from typing import Any, Dict

import msgpack  # type: ignore[import-untyped]

GRAPH_STATIC_VERSION = 1
SNAPSHOT_DELTA_VERSION = 1


def pack_graph_static(data: Dict[str, Any]) -> bytes:
    """Return msgpack-encoded ``GraphStatic`` message."""
    payload = {"type": "GraphStatic", "v": GRAPH_STATIC_VERSION, **data}
    return msgpack.packb(payload, use_bin_type=True, use_single_float=True)


def unpack_graph_static(raw: bytes) -> Dict[str, Any]:
    """Decode a ``GraphStatic`` message ensuring version compatibility."""
    msg = msgpack.unpackb(raw, raw=False)
    if msg.get("type") != "GraphStatic":
        raise ValueError("expected type 'GraphStatic'")
    if "v" not in msg:
        raise ValueError("missing 'v' field")
    if msg["v"] != GRAPH_STATIC_VERSION:
        raise ValueError(f"unsupported GraphStatic version: {msg['v']}")
    msg.pop("type", None)
    msg.pop("v", None)
    return msg


def pack_snapshot_delta(data: Dict[str, Any]) -> bytes:
    """Return msgpack-encoded ``SnapshotDelta`` message."""
    payload = {"type": "SnapshotDelta", "v": SNAPSHOT_DELTA_VERSION, **data}
    return msgpack.packb(payload, use_bin_type=True, use_single_float=True)


def unpack_snapshot_delta(raw: bytes) -> Dict[str, Any]:
    """Decode a ``SnapshotDelta`` message ensuring version compatibility."""
    msg = msgpack.unpackb(raw, raw=False)
    if msg.get("type") != "SnapshotDelta":
        raise ValueError("expected type 'SnapshotDelta'")
    if "v" not in msg:
        raise ValueError("missing 'v' field")
    if msg["v"] != SNAPSHOT_DELTA_VERSION:
        raise ValueError(f"unsupported SnapshotDelta version: {msg['v']}")
    msg.pop("type", None)
    msg.pop("v", None)
    return msg
