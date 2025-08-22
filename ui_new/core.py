"""Core loop wiring WebSocket messages to the QML graph view."""

from __future__ import annotations

import asyncio
from typing import Any

from .ipc import Client
from .state import (
    Store,
    TelemetryModel,
    ExperimentModel,
    ReplayModel,
    LogsModel,
    DOEModel,
    GAModel,
    MCTSModel,
)


async def run(
    url: str,
    view,
    telemetry: TelemetryModel,
    experiment: ExperimentModel,
    replay: ReplayModel,
    logs: LogsModel,
    store: Store,
    doe: DOEModel,
    ga: GAModel,
    mcts: MCTSModel,
    window: Any,
    token: str | None = None,
) -> None:
    """Connect to ``url`` and forward graph updates to the view and models.

    ``view`` must expose :meth:`set_graph` and :meth:`apply_delta` methods.
    Other models receive telemetry, experiment status, replay progress and log
    entries for display in QML panels. ``token`` is forwarded to the server for
    simple session authentication. ``window`` controls overall UI enabling and
    is disabled on disconnect. A 2-second heartbeat detects dead engines and
    drops the connection after missed pongs.
    """

    window.controlsEnabled = False
    client = Client(url, token, ping_interval=2.0)
    await client.connect()
    window.controlsEnabled = True

    loop = asyncio.get_running_loop()
    experiment.set_client(client)
    replay.set_client(client)
    store.set_client(client)
    doe.set_client(client)
    ga.set_client(client, loop)
    mcts.set_client(client, loop)

    try:
        while True:
            msg = await client.receive()

            mtype = msg.get("type")
            if mtype == "GraphStatic":
                static = {k: v for k, v in msg.items() if k not in {"type", "v"}}
                store.set_static(static)
                nodes = static.get("node_positions", [])
                edges = static.get("edges", [])
                labels = static.get("node_labels")
                colors = static.get("node_colors")
                flags = static.get("node_flags")
                view.set_graph(nodes, edges, labels, colors, flags)
                telemetry.update_counts(len(nodes), len(edges))
                continue
            if mtype == "SnapshotDelta":
                delta = {k: v for k, v in msg.items() if k not in {"type", "v"}}
                store.apply_delta(delta)
                view.apply_delta(delta)
                telemetry.update_counts(len(view._nodes), len(view._edges))
                counters = delta.get("counters")
                invariants = delta.get("invariants")
                depth = delta.get("depth")
                label = "depth"
                if depth is None:
                    depth = delta.get("frame")
                    label = "frame"
                if counters or invariants or depth is not None:
                    telemetry.record(counters, invariants, depth=depth, label=label)
                continue

            if mtype == "ExperimentStatus":
                experiment.update(msg.get("status", ""), msg.get("residual", 0.0))
                doe.handle_status(msg)
                ga.handle_status(msg)
                mcts.handle_status(msg)
                continue

            if mtype == "ReplayProgress":
                progress = msg.get("progress")
                if progress is not None:
                    replay.update_progress(float(progress))
                continue

            if mtype == "LogEntry":
                entry = msg.get("entry")
                if entry is not None:
                    logs.add_entry(str(entry))
    finally:
        view.editable = False
        window.controlsEnabled = False
        experiment.set_client(None)
        replay.set_client(None)
        store.set_client(None)
        doe.set_client(None)
        ga.set_client(None)
        mcts.set_client(None)
