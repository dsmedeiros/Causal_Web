"""WebSocket client using msgpack for communication."""

from __future__ import annotations

from typing import Any, Deque, Dict, Optional

import asyncio
from collections import deque

import msgpack
import websockets


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
        self._backlog: Deque[Dict[str, Any]] = deque()

    async def connect(self) -> None:
        """Open the WebSocket connection and perform the hello handshake."""
        self.connection = await websockets.connect(self.url)
        await self.send({"type": "Hello", "token": self.token})
        self._ping_task = asyncio.create_task(self._heartbeat())

    async def _heartbeat(self) -> None:
        """Send periodic ``Ping`` messages and close after missed pongs."""
        misses = 0
        try:
            while self.connection:
                await asyncio.sleep(self.ping_interval)
                if not self.connection:
                    break
                await self.send({"type": "Ping"})
                try:
                    while True:
                        data = await asyncio.wait_for(
                            self.connection.recv(), timeout=self.ping_interval
                        )
                        msg = msgpack.unpackb(data, raw=False)
                        if msg.get("type") == "Pong":
                            misses = 0
                            break
                        self._backlog.append(msg)
                except (asyncio.TimeoutError, websockets.ConnectionClosed):
                    misses += 1
                    if misses >= 2 and self.connection:
                        await self.connection.close()
                        self.connection = None
                    continue
        except websockets.ConnectionClosed:
            pass

    async def send(self, message: Dict[str, Any]) -> None:
        """Serialize and send ``message``."""
        if not self.connection:
            raise RuntimeError("Client not connected")
        packed = msgpack.packb(message, use_bin_type=True)
        await self.connection.send(packed)

    async def receive(self) -> Dict[str, Any]:
        """Receive and deserialize a message from the server."""
        if self._backlog:
            return self._backlog.popleft()
        if not self.connection:
            raise RuntimeError("Client not connected")
        data = await self.connection.recv()
        return msgpack.unpackb(data, raw=False)

    async def drop_pending(self, mtype: str) -> None:
        """Discard queued messages of ``mtype`` leaving others untouched."""

        if not self.connection:
            return
        while True:
            try:
                data = await asyncio.wait_for(self.connection.recv(), timeout=0)
            except asyncio.TimeoutError:
                break
            msg = msgpack.unpackb(data, raw=False)
            if msg.get("type") == mtype:
                continue
            self._backlog.append(msg)

    async def close(self) -> None:
        """Close the WebSocket connection."""
        if self._ping_task is not None:
            self._ping_task.cancel()
        if self.connection is not None:
            await self.connection.close()
            self.connection = None
