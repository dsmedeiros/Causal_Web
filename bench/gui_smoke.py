"""Lightweight GUI apply_delta performance smoke test."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from PySide6.QtGui import QGuiApplication

from ui_new.graph.GraphView import GraphView
from machine import machine_info


def _sample_graph(
    n_nodes: int,
) -> tuple[list[tuple[float, float]], list[tuple[int, int]]]:
    """Return simple cyclic graph of ``n_nodes`` positions and edges."""

    nodes = [(float(i), 0.0) for i in range(n_nodes)]
    edges = [(i, (i + 1) % n_nodes) for i in range(n_nodes)]
    return nodes, edges


def bench_apply_delta(n_nodes: int = 5000) -> dict[str, float | int]:
    """Measure time to apply a delta update for ``n_nodes`` nodes."""

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QGuiApplication.instance() or QGuiApplication([])
    view = GraphView()
    nodes, edges = _sample_graph(n_nodes)
    view.set_graph(nodes, edges)
    delta = {"node_positions": {i: (x + 1.0, 0.0) for i, (x, _) in enumerate(nodes)}}
    # Warm up and measure multiple times, keeping the fastest sample.
    timings = []
    for _ in range(3):
        t0 = time.perf_counter()
        view.apply_delta(delta)
        t1 = time.perf_counter()
        timings.append((t1 - t0) * 1000.0)
    app.quit()
    return {"nodes": n_nodes, "apply_delta_ms": min(timings)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", type=int, default=5000)
    parser.add_argument(
        "--output", type=Path, help="Optional path to write JSON results."
    )
    args = parser.parse_args()
    result = bench_apply_delta(args.nodes)
    result["machine"] = machine_info()
    if args.output:
        args.output.write_text(json.dumps(result))
    print(json.dumps(result))


if __name__ == "__main__":  # pragma: no cover
    main()
