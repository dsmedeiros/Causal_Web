"""WebSocket client using msgpack for communication."""

from __future__ import annotations

from typing import Any, Deque, Dict, Optional

import asyncio
from collections import deque
import contextlib

import msgpack
import websockets
from websockets.exceptions import ConnectionClosedError


class ConnectError(Exception):
    """Raised when the client fails to establish a connection."""


class Client:
    """Simple WebSocket client using msgpack for message encoding."""

    def __init__(
        self, url: str, token: str | None = None, ping_interval: float = 30.0
    ) -> None:
        """Initialize the client.

        Parameters
        ----------
        url:
            WebSocket server URL.
        token:
            Optional session token required by the server.
        ping_interval:
            Seconds between heartbeat pings.
        """
        self.url = url
        self.token = token or ""
        self.ping_interval = ping_interval
        self.connection: Optional[websockets.WebSocketClientProtocol] = None
        self._ping_task: Optional[asyncio.Task] = None
        self._recv_task: Optional[asyncio.Task] = None
        self._backlog: Deque[Dict[str, Any]] = deque()
        self._backlog_cv = asyncio.Condition()

    async def connect(self) -> None:
        """Open the WebSocket connection and perform the hello handshake."""
        try:
            self.connection = await websockets.connect(self.url)
            await self.send({"type": "Hello", "token": self.token})
            data = await self.connection.recv()
            msg = msgpack.unpackb(data, raw=False)
            if msg.get("type") != "Hello":
                raise ConnectError("handshake failed")
        except (OSError, ConnectionClosedError) as e:
            if self.connection is not None:
                await self.connection.close()
                self.connection = None
            raise ConnectError(str(e)) from e
        self._recv_task = asyncio.create_task(self._receiver())
        self._ping_task = asyncio.create_task(self._heartbeat())

    async def _receiver(self) -> None:
        """Background task pulling messages from the connection."""
        assert self.connection is not None
        try:
            while self.connection is not None:
                data = await self.connection.recv()
                msg = msgpack.unpackb(data, raw=False)
                async with self._backlog_cv:
                    self._backlog.append(msg)
                    self._backlog_cv.notify()
        except (asyncio.CancelledError, websockets.ConnectionClosed):
            pass
        finally:
            if self.connection is not None:
                await self.connection.close()
                self.connection = None

    async def _heartbeat(self) -> None:
        """Issue periodic ping frames and close when unanswered."""
        try:
            while self.connection is not None:
                try:
                    waiter = self.connection.ping()
                    await asyncio.wait_for(waiter, timeout=self.ping_interval)
                except (asyncio.TimeoutError, websockets.ConnectionClosed):
                    if self.connection is not None:
                        await self.connection.close()
                    self.connection = None
                    break
                await asyncio.sleep(self.ping_interval)
        except asyncio.CancelledError:
            pass

    async def send(self, message: Dict[str, Any]) -> None:
        """Serialize and send ``message``."""
        if self.connection is None:
            raise RuntimeError("Client not connected")
        packed = msgpack.packb(message, use_bin_type=True)
        await self.connection.send(packed)

    async def receive(self) -> Dict[str, Any]:
        """Receive and deserialize a message from the server."""
        async with self._backlog_cv:
            while not self._backlog:
                if self.connection is None:
                    raise RuntimeError("Client not connected")
                await self._backlog_cv.wait()
            return self._backlog.popleft()

    async def drop_pending(self, mtype: str) -> None:
        """Discard queued messages of ``mtype`` leaving others untouched."""

        async with self._backlog_cv:
            self._backlog = deque(
                msg for msg in self._backlog if msg.get("type") != mtype
            )

    async def close(self) -> None:
        """Close the WebSocket connection."""
        if self._ping_task is not None:
            self._ping_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._ping_task
            self._ping_task = None
        if self._recv_task is not None:
            self._recv_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._recv_task
            self._recv_task = None
        if self.connection is not None:
            await self.connection.close()
            self.connection = None
