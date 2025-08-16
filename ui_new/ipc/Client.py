"""WebSocket client using msgpack for communication."""

from __future__ import annotations

from typing import Any, Dict, Optional

import msgpack
import websockets


class Client:
    """Simple WebSocket client using msgpack for message encoding."""

    def __init__(self, url: str) -> None:
        """Initialize the client with the server ``url``."""
        self.url = url
        self.connection: Optional[websockets.WebSocketClientProtocol] = None

    async def connect(self) -> None:
        """Open the WebSocket connection."""
        self.connection = await websockets.connect(self.url)

    async def send(self, message: Dict[str, Any]) -> None:
        """Serialize and send ``message``."""
        if not self.connection:
            raise RuntimeError("Client not connected")
        packed = msgpack.packb(message, use_bin_type=True)
        await self.connection.send(packed)

    async def receive(self) -> Dict[str, Any]:
        """Receive and deserialize a message from the server."""
        if not self.connection:
            raise RuntimeError("Client not connected")
        data = await self.connection.recv()
        return msgpack.unpackb(data, raw=False)

    async def close(self) -> None:
        """Close the WebSocket connection."""
        if self.connection is not None:
            await self.connection.close()
            self.connection = None
