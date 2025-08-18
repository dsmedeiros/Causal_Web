"""WebSocket server for publishing engine state deltas.

This module exposes a :class:`DeltaBus` ring buffer and a ``serve`` coroutine
that publishes graph metadata and snapshot deltas to connected clients. Payloads
are encoded with `msgpack` and versioned via a ``type`` and ``v`` field.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, Dict, Set

import msgpack
import websockets


class DeltaBus:
    """Double buffer retaining only the most recent snapshot delta.

    The engine places freshly computed deltas into the bus. Consumers read the
    latest value without worrying about intermediate frames.
    """

    def __init__(self) -> None:
        self._slots = [None, None]
        self._idx = 0

    def put(self, delta: Dict[str, Any]) -> None:
        """Store ``delta`` and swap buffers so it becomes the latest."""
        self._idx ^= 1
        self._slots[self._idx] = delta

    def latest(self) -> Dict[str, Any] | None:
        """Return the newest snapshot delta if available."""
        return self._slots[self._idx]


async def serve(
    bus: DeltaBus,
    adapter: Any,
    host: str = "127.0.0.1",
    port: int = 8765,
    allow_multi: bool | None = None,
    session_token: str | None = None,
) -> None:
    """Start a WebSocket server publishing engine deltas.

    Parameters
    ----------
    bus:
        :class:`DeltaBus` used to retain the latest snapshot delta.
    adapter:
        Object providing ``graph_static``, ``snapshot_delta``,
        ``handle_control`` and ``experiment_status`` callables.
    host, port:
        Network location on which to serve the WebSocket.
    allow_multi:
        Whether to permit multiple simultaneous clients. If ``None`` the value
        is read from the ``CW_ALLOW_MULTI`` environment variable and defaults to
        ``False``.
    session_token:
        Optional token expected from clients on connect.
    """

    if allow_multi is None:
        allow_multi = os.getenv("CW_ALLOW_MULTI", "false").lower() in {
            "1",
            "true",
            "yes",
        }

    clients: Set[websockets.WebSocketServerProtocol] = set()

    async def handler(ws: websockets.WebSocketServerProtocol) -> None:
        """Handle a single client connection."""

        if not allow_multi and clients:
            await ws.close(reason="single client only")
            return

        raw = await ws.recv()
        msg = msgpack.unpackb(raw, raw=False)
        token = msg.get("token", "")
        if session_token and token != session_token:
            await ws.close(reason="unauthorized")
            return

        clients.add(ws)
        try:
            graph_static = {"type": "GraphStatic", "v": 1, **adapter.graph_static()}
            await ws.send(msgpack.packb(graph_static))
            async for raw in ws:
                msg = msgpack.unpackb(raw, raw=False)
                cmd = msg.get("cmd")
                if cmd == "pull":
                    latest = bus.latest()
                    if latest is not None:
                        payload = {"type": "SnapshotDelta", "v": 1, **latest}
                        await ws.send(msgpack.packb(payload))
                else:
                    result = adapter.handle_control(msg)
                    if result:
                        await ws.send(msgpack.packb(result))
        finally:
            clients.remove(ws)

    async with websockets.serve(handler, host, port):
        while True:
            delta = adapter.snapshot_delta()
            if delta is not None:
                bus.put(delta)
                if clients:
                    notify = {"type": "DeltaReady", "v": 1, "frame": delta.get("frame")}
                    await asyncio.gather(
                        *[ws.send(msgpack.packb(notify)) for ws in list(clients)]
                    )
            if hasattr(adapter, "experiment_status"):
                status = adapter.experiment_status()
                if status and clients:
                    payload = {"type": "ExperimentStatus", "v": 1, **status}
                    await asyncio.gather(
                        *[ws.send(msgpack.packb(payload)) for ws in list(clients)]
                    )
            if hasattr(adapter, "replay_progress"):
                progress = adapter.replay_progress()
                if progress is not None and clients:
                    payload = {
                        "type": "ReplayProgress",
                        "v": 1,
                        "progress": float(progress),
                    }
                    await asyncio.gather(
                        *[ws.send(msgpack.packb(payload)) for ws in list(clients)]
                    )
            await asyncio.sleep(0)
