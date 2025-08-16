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
)


async def run(
    url: str,
    view,
    telemetry: TelemetryModel,
    experiment: ExperimentModel,
    replay: ReplayModel,
    logs: LogsModel,
) -> None:
    """Connect to ``url`` and forward graph updates to the view and models.

    ``view`` must expose :meth:`set_graph` and :meth:`apply_delta` methods.
    Other models receive telemetry, experiment status, replay progress and log
    entries for display in QML panels.
    """

    client = Client(url)
    store = Store()
    await client.connect()

    experiment.set_client(client)
    replay.set_client(client)

    msg = await client.receive()
    if msg.get("type") == "GraphStatic":
        static = {k: v for k, v in msg.items() if k not in {"type", "v"}}
        store.set_static(static)
        nodes = static.get("node_positions", [])
        edges = static.get("edges", [])
        labels = static.get("node_labels")
        colors = static.get("node_colors")
        flags = static.get("node_flags")
        view.set_graph(nodes, edges, labels, colors, flags)
        telemetry.update_counts(len(nodes), len(edges))

    while True:
        msg = await client.receive()

        mtype = msg.get("type")
        if mtype == "SnapshotDelta":
            delta = {k: v for k, v in msg.items() if k not in {"type", "v"}}
            store.apply_delta(delta)
            view.apply_delta(delta)
            telemetry.update_counts(len(view._nodes), len(view._edges))
            continue

        if mtype == "ExperimentStatus":
            experiment.update(msg.get("status", ""), msg.get("residual", 0.0))
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
