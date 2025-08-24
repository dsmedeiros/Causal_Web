import asyncio
import threading
import time

from Causal_Web.engine.stream.server import DeltaBus, serve
from ui_new import core
from ui_new.auth import resolve_connection_info
from ui_new.state import (
    Store,
    TelemetryModel,
    ExperimentModel,
    ReplayModel,
    LogsModel,
    DOEModel,
    GAModel,
    MCTSModel,
    PolicyModel,
)


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


def test_gui_reads_bundle_and_receives_graph(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    bus = DeltaBus()
    bus.put({"frame": 0})
    adapter = Adapter()
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=_start_loop, args=(loop,), daemon=True)
    thread.start()
    server_future = asyncio.run_coroutine_threadsafe(serve(bus, adapter), loop)
    time.sleep(0.1)
    url, token = resolve_connection_info()
    telemetry = TelemetryModel()
    experiment = ExperimentModel()
    replay = ReplayModel()
    logs = LogsModel()
    store = Store()
    doe = DOEModel()
    ga = GAModel()
    mcts = MCTSModel()
    policy = PolicyModel()
    gui_future = asyncio.run_coroutine_threadsafe(
        core.run(
            url,
            View(),
            telemetry,
            experiment,
            replay,
            logs,
            store,
            doe,
            ga,
            mcts,
            policy,
            Window(),
            token=token,
        ),
        loop,
    )
    time.sleep(0.5)
    assert store.graph_static
    gui_future.cancel()
    server_future.cancel()
    loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=1)
