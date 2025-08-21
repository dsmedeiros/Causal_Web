import os
import asyncio
import threading
import time

import pytest


@pytest.mark.skipif(not os.getenv("CW_GUI_SMOKE"), reason="CW_GUI_SMOKE not set")
def test_engine_gui_smoke():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    from PySide6.QtGui import QGuiApplication
    from Causal_Web.engine.engine_v2.adapter import EngineAdapter
    from Causal_Web.engine.stream.server import DeltaBus, serve
    from ui_new import core
    from ui_new.state import (
        Store,
        TelemetryModel,
        MetersModel,
        ExperimentModel,
        ReplayModel,
        LogsModel,
        DOEModel,
        GAModel,
        CompareModel,
    )

    app = QGuiApplication([])

    adapter = EngineAdapter()
    graph = {"params": {"W0": 2}, "nodes": [{"id": "0", "rho_mean": 0.0}], "edges": []}
    adapter.build_graph(graph)

    bus = DeltaBus()
    bus.put({"frame": 0})

    loop = asyncio.new_event_loop()

    def run_loop() -> None:
        asyncio.set_event_loop(loop)
        loop.run_forever()

    thread = threading.Thread(target=run_loop, daemon=True)
    thread.start()

    server_future = asyncio.run_coroutine_threadsafe(serve(bus, adapter), loop)

    telemetry = TelemetryModel()
    meters = MetersModel()
    experiment = ExperimentModel()
    replay = ReplayModel()
    logs = LogsModel()
    store = Store()
    doe = DOEModel()
    ga = GAModel()
    compare = CompareModel()

    class View:
        def set_graph(self, *args, **kwargs):
            pass

        def apply_delta(self, *args, **kwargs):
            pass

    class Window:
        controlsEnabled = False

    gui_future = asyncio.run_coroutine_threadsafe(
        core.run(
            "ws://127.0.0.1:8765",
            View(),
            telemetry,
            experiment,
            replay,
            logs,
            store,
            doe,
            ga,
            Window(),
            token=None,
        ),
        loop,
    )

    time.sleep(0.5)
    assert "nodes" in store.graph_static, "Did not receive graph data"

    gui_future.cancel()
    server_future.cancel()
    loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=1)
    app.quit()
