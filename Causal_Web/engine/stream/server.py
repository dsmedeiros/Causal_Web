"""WebSocket server for publishing engine state deltas.

This module exposes a :class:`DeltaBus` ring buffer and a ``serve`` coroutine
that publishes graph metadata and snapshot deltas to connected clients. Payloads
are encoded with `msgpack` and versioned via a ``type`` and ``v`` field.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import secrets
import time
from typing import Any, Dict, Set

import msgpack
import websockets

from .auth import write_session_bundle


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


async def _broadcast(
    payload: bytes, clients: Set[websockets.WebSocketServerProtocol]
) -> None:
    """Send ``payload`` to all ``clients`` removing stale sockets."""

    stale: list[websockets.WebSocketServerProtocol] = []
    for ws in list(clients):
        try:
            await ws.send(payload)
        except Exception:
            stale.append(ws)
    for ws in stale:
        clients.discard(ws)


async def publisher(
    bus: DeltaBus, clients: Set[websockets.WebSocketServerProtocol], poll_hz: int = 120
) -> None:
    """Publish snapshot deltas from ``bus`` to connected ``clients``.

    Parameters
    ----------
    bus:
        :class:`DeltaBus` providing the latest snapshot delta.
    clients:
        Set of websocket connections to broadcast deltas to.
    poll_hz:
        Frequency in Hertz at which to poll the bus when idle.
    """

    last_frame: Any = None
    idle_sleep = 1.0 / poll_hz
    while True:
        delta = bus.latest()
        frame = delta.get("frame") if delta else None
        if delta is not None and frame != last_frame:
            payload = msgpack.packb(
                {"type": "SnapshotDelta", "v": 1, **delta},
                use_bin_type=True,
                use_single_float=True,
            )
            await _broadcast(payload, clients)
            last_frame = frame
            await asyncio.sleep(0)
        else:
            await asyncio.sleep(idle_sleep)


async def serve(
    bus: DeltaBus,
    adapter: Any,
    host: str = "127.0.0.1",
    port: int = 8765,
    allow_multi: bool | None = None,
    session_token: str | None = None,
    session_file: str | None = None,
    session_ttl: int | None = None,
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
        Optional token expected from clients on connect. If ``None`` a random
        token is generated and recorded in the session bundle.
    session_file:
        Path to the JSON session bundle written for GUI auto-discovery.
    session_ttl:
        Lifetime of the session token in seconds. Defaults to ``3600``.
    """

    if allow_multi is None:
        allow_multi = os.getenv("CW_ALLOW_MULTI", "false").lower() in {
            "1",
            "true",
            "yes",
        }

    host = os.getenv("CW_WS_HOST", host)
    port = int(os.getenv("CW_WS_PORT", str(port)))
    session_file = session_file or os.getenv("CW_SESSION_FILE")
    session_token = session_token or os.getenv("CW_SESSION_TOKEN")
    session_ttl = session_ttl or int(os.getenv("CW_SESSION_TTL", "3600"))

    if session_token is None:
        session_token = secrets.token_urlsafe(16)

    bundle, bundle_path = write_session_bundle(
        host, port, session_token, session_ttl, session_file
    )
    print(f"CW session token: {session_token} (bundle: {bundle_path})")
    expires_at = bundle["expires_at"]

    clients: Set[websockets.WebSocketServerProtocol] = set()
    primary: websockets.WebSocketServerProtocol | None = None
    pending: websockets.WebSocketServerProtocol | None = None

    publisher_task = asyncio.create_task(publisher(bus, clients))

    async def _send_roles() -> None:
        nonlocal primary
        for ws in list(clients):
            role = "controller" if ws is primary else "spectator"
            try:
                await ws.send(
                    msgpack.packb(
                        {"type": "Role", "v": 1, "role": role},
                        use_bin_type=True,
                        use_single_float=True,
                    )
                )
            except Exception:
                clients.discard(ws)
                if ws is primary:
                    primary = next(iter(clients), None)

    async def handler(ws: websockets.WebSocketServerProtocol) -> None:
        """Handle a single client connection.

        The first client completing the hello handshake becomes the controller.
        Subsequent clients are spectators unless ``allow_multi`` is ``False`` in
        which case they are rejected. Spectators may issue ``RequestControl`` and
        the current controller can yield via ``TransferControl``.
        """

        nonlocal primary, pending

        raw = await ws.recv()
        msg = msgpack.unpackb(raw, raw=False)
        if msg.get("type") != "Hello":
            await ws.close(reason="handshake required")
            return
        token = msg.get("token", "")
        now = time.time()
        if now >= expires_at:
            await ws.close(reason="session expired")
            return
        if session_token and token != session_token:
            await ws.close(reason="unauthorized")
            return

        if primary is None:
            primary = ws
        elif not allow_multi:
            await ws.close(reason="single client only")
            return

        clients.add(ws)
        try:
            await ws.send(
                msgpack.packb(
                    {"type": "Hello", "v": 1}, use_bin_type=True, use_single_float=True
                )
            )
            graph_static = {"type": "GraphStatic", "v": 1, **adapter.graph_static()}
            await ws.send(
                msgpack.packb(graph_static, use_bin_type=True, use_single_float=True)
            )
            await _send_roles()
            async for raw in ws:
                msg = msgpack.unpackb(raw, raw=False)
                mtype = msg.get("type")
                if mtype == "Ping":
                    await ws.send(
                        msgpack.packb(
                            {"type": "Pong", "v": 1},
                            use_bin_type=True,
                            use_single_float=True,
                        )
                    )
                    continue
                if mtype == "RequestControl" and ws is not primary:
                    pending = ws
                    if primary is not None:
                        try:
                            await primary.send(
                                msgpack.packb(
                                    {"type": "ControlRequest", "v": 1},
                                    use_bin_type=True,
                                    use_single_float=True,
                                )
                            )
                        except Exception:
                            clients.discard(primary)
                            primary = None
                    continue
                if mtype == "TransferControl" and ws is primary:
                    if pending in clients:
                        primary = pending
                        pending = None
                        await _send_roles()
                    else:
                        await ws.send(
                            msgpack.packb(
                                {"type": "Error", "v": 1, "reason": "no requester"},
                                use_bin_type=True,
                                use_single_float=True,
                            )
                        )
                    continue
                cmd = msg.get("cmd")
                if cmd == "pull":
                    latest = bus.latest()
                    if latest is not None:
                        payload = {"type": "SnapshotDelta", "v": 1, **latest}
                        await ws.send(
                            msgpack.packb(
                                payload, use_bin_type=True, use_single_float=True
                            )
                        )
                elif ws is not primary:
                    await ws.send(
                        msgpack.packb(
                            {"type": "Error", "v": 1, "reason": "spectator"},
                            use_bin_type=True,
                            use_single_float=True,
                        )
                    )
                else:
                    result = adapter.handle_control(msg)
                    if result:
                        await ws.send(
                            msgpack.packb(
                                result, use_bin_type=True, use_single_float=True
                            )
                        )
        finally:
            clients.discard(ws)
            if ws is primary:
                primary = pending if pending in clients else next(iter(clients), None)
                pending = None
            elif ws is pending:
                pending = None
            await _send_roles()

    try:
        async with websockets.serve(handler, host, port):
            idle_sleep = 1.0 / 120
            while True:
                delta = adapter.snapshot_delta()
                if delta is not None:
                    bus.put(delta)

                if hasattr(adapter, "experiment_status"):
                    status = adapter.experiment_status()
                    if status and clients:
                        payload = msgpack.packb(
                            {"type": "ExperimentStatus", "v": 1, **status},
                            use_bin_type=True,
                            use_single_float=True,
                        )
                        await _broadcast(payload, clients)

                if hasattr(adapter, "replay_progress"):
                    progress = adapter.replay_progress()
                    if progress is not None and clients:
                        payload = msgpack.packb(
                            {
                                "type": "ReplayProgress",
                                "v": 1,
                                "progress": float(progress),
                            },
                            use_bin_type=True,
                            use_single_float=True,
                        )
                        await _broadcast(payload, clients)

                if delta is None:
                    await asyncio.sleep(idle_sleep)
                else:
                    await asyncio.sleep(0)
    finally:
        publisher_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await publisher_task
