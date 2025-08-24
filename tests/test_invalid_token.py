import asyncio
import threading
import time

import pytest

from Causal_Web.engine.stream.server import DeltaBus, serve
from ui_new.ipc import Client, ConnectError


class View:
    def set_graph(self, *args, **kwargs):
        pass

    def apply_delta(self, *args, **kwargs):
        pass


class Window:
    controlsEnabled = False


def _start_loop(loop):
    asyncio.set_event_loop(loop)
    loop.run_forever()


class Adapter:
    def graph_static(self):
        return {"node_positions": [], "edges": []}


def test_gui_rejects_wrong_token(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    bus = DeltaBus()
    bus.put({"frame": 0})
    adapter = Adapter()
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=_start_loop, args=(loop,), daemon=True)
    thread.start()
    server_future = asyncio.run_coroutine_threadsafe(
        serve(bus, adapter, session_token="right"), loop
    )
    time.sleep(0.1)
    client = Client("ws://127.0.0.1:8765", token="wrong")
    with pytest.raises(ConnectError):
        asyncio.run(client.connect())
    server_future.cancel()
    loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=1)
