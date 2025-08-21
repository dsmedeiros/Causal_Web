"""Benchmark engine step throughput.

This script builds a small sample graph, advances the engine for a number
of steps and reports the achieved steps per second as JSON.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from Causal_Web.engine.engine_v2.adapter import EngineAdapter
from machine import machine_info


def sample_graph(n: int = 100) -> dict:
    """Return a simple cyclic graph with ``n`` nodes."""

    nodes = [{"id": str(i), "window_len": 1} for i in range(n)]
    edges = [
        {"from": str(i), "to": str((i + 1) % n), "delay": 1.0, "density": 0.0}
        for i in range(n)
    ]
    return {"nodes": nodes, "edges": edges, "params": {"W0": 1}}


def bench_engine(n_steps: int = 20_000) -> dict:
    """Return steps per second achieved over ``n_steps``."""

    engine = EngineAdapter()
    engine.build_graph(sample_graph())
    t0 = time.perf_counter()
    for _ in range(n_steps):
        engine.step()
    t1 = time.perf_counter()
    return {"steps": n_steps, "steps_per_sec": n_steps / (t1 - t0)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument(
        "--output", type=Path, help="Optional path to write JSON results."
    )
    parser.add_argument(
        "--machine",
        type=str,
        help="Optional machine notes for the JSON output.",
    )
    args = parser.parse_args()
    result = bench_engine(args.steps)
    result["machine"] = machine_info()
    if args.machine:
        result["machine"]["notes"] = args.machine
    if args.output:
        args.output.write_text(json.dumps(result))
    print(json.dumps(result))


if __name__ == "__main__":  # pragma: no cover
    main()
