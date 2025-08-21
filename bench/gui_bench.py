"""Benchmark GUI frame rendering rate.

This script launches a minimal Qt Quick window containing the
``GraphView`` and measures the number of frames rendered over a fixed
duration. Graph size, anti-aliasing and label rendering are configurable
so different scenarios can be profiled. Results are reported as JSON for
ingestion by CI.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from PySide6.QtCore import QTimer
from PySide6.QtGui import QGuiApplication
from PySide6.QtQml import QQmlApplicationEngine
from PySide6.QtQuick import QQuickItem

from ui_new.graph import GraphView  # register QML module
from ui_new.state import MetersModel
from machine import machine_info


def _sample_graph(
    n_nodes: int = 100,
) -> tuple[list[tuple[float, float]], list[tuple[int, int]]]:
    """Return node positions and edges for a simple cyclic graph."""

    nodes = [(float(i), 0.0) for i in range(n_nodes)]
    edges = [(i, (i + 1) % n_nodes) for i in range(n_nodes)]
    return nodes, edges


def bench_gui(
    duration: float = 5.0,
    n_nodes: int = 100,
    aa: bool = True,
    labels: bool = True,
) -> dict[str, float | int | bool]:
    """Measure frames per second over ``duration`` seconds.

    Args:
        duration: Measurement window in seconds.
        n_nodes: Number of nodes in the sample graph.
        aa: Enable anti-aliasing.
        labels: Render node labels.
    """

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QGuiApplication([])
    engine = QQmlApplicationEngine()
    meters = MetersModel()

    qml = b"""
import QtQuick 2.15
import QtQuick.Window 2.15
import CausalGraph 1.0

Window {
    id: root
    width: 800
    height: 600
    visible: true
    GraphView {
        id: graphView
        objectName: "graphView"
        anchors.fill: parent
    }
}
"""
    engine.loadData(qml)
    if not engine.rootObjects():
        raise RuntimeError("Failed to load QML window")
    root = engine.rootObjects()[0]
    view = root.findChild(QQuickItem, "graphView")
    nodes, edges = _sample_graph(n_nodes)
    view.set_graph(nodes, edges)
    view.frameRendered.connect(meters.frame_drawn)
    view.antialiasThreshold = 0.0 if aa else 2.0
    view.labelThreshold = 0.0 if labels else 2.0

    timer = QTimer()
    timer.timeout.connect(view.update)
    timer.start(0)

    QTimer.singleShot(int(duration * 1000), app.quit)
    t0 = time.perf_counter()
    app.exec()
    t1 = time.perf_counter()
    elapsed = t1 - t0
    fps = meters.frame / elapsed if elapsed > 0 else 0.0
    return {
        "duration": elapsed,
        "fps": fps,
        "nodes": n_nodes,
        "aa": aa,
        "labels": labels,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--nodes", type=int, default=100)
    parser.add_argument(
        "--aa",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable or disable anti-aliasing.",
    )
    parser.add_argument(
        "--labels",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Render node labels.",
    )
    parser.add_argument("--target-fps", type=float, help="Configured target FPS")
    parser.add_argument(
        "--machine", type=str, help="Additional machine notes for the JSON output."
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to write JSON results.",
    )
    args = parser.parse_args()
    result = bench_gui(args.duration, args.nodes, args.aa, args.labels)
    result["machine"] = machine_info()
    if args.target_fps is not None:
        result["target_fps"] = args.target_fps
    if args.machine:
        result["machine"]["notes"] = args.machine
    if args.output:
        args.output.write_text(json.dumps(result))
    print(json.dumps(result))


if __name__ == "__main__":  # pragma: no cover
    main()
