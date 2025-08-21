"""Headless renderer composing snapshot logs into an MP4 video.

This utility avoids the Qt GUI and instead uses ``matplotlib`` and
``imageio`` to draw recorded graphs and deltas.  ``GraphStatic`` data seeds the
initial layout while subsequent ``SnapshotDelta`` entries update node
positions and edges.  Each frame is rendered offscreen and appended to an MP4
writer.

Example
-------
```
python -m tools.video_export graph_static.json delta_log.jsonl out.mp4
```
"""

from __future__ import annotations

import argparse
import json
from typing import Iterable, Tuple

import imageio
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _draw_frame(
    ax: "plt.Axes",
    positions: dict[int, Tuple[float, float]],
    edges: list[Tuple[int, int]],
) -> np.ndarray:
    """Render the current graph state to an RGB array."""

    ax.clear()
    ax.set_aspect("equal")
    ax.set_axis_off()
    xs = [p[0] for _, p in sorted(positions.items())]
    ys = [p[1] for _, p in sorted(positions.items())]
    ax.scatter(xs, ys, c="white", s=10)
    for a, b in edges:
        x0, y0 = positions[a]
        x1, y1 = positions[b]
        ax.plot([x0, x1], [y0, y1], color="gray", linewidth=0.5)
    ax.figure.canvas.draw()
    img = np.frombuffer(ax.figure.canvas.tostring_rgb(), dtype=np.uint8)
    return img.reshape(ax.figure.canvas.get_width_height()[::-1] + (3,))


def render_video(
    graph_static_path: str,
    delta_log_path: str,
    output_path: str,
    fps: int = 30,
    max_frames: int | None = None,
) -> None:
    """Render snapshot logs to an MP4 file.

    Parameters
    ----------
    graph_static_path:
        Path to ``GraphStatic`` JSON file.
    delta_log_path:
        Path to newline-delimited ``SnapshotDelta`` log.
    output_path:
        Destination ``.mp4`` file.
    fps:
        Frames per second of the output video.
    max_frames:
        Optional cap on the number of frames rendered.
    """

    with open(graph_static_path, "r", encoding="utf-8") as fh:
        gs = json.load(fh)
    positions = {i: tuple(p) for i, p in enumerate(gs.get("node_positions", []))}
    edges = [tuple(e) for e in gs.get("edges", [])]

    writer = imageio.get_writer(output_path, fps=fps)
    fig, ax = plt.subplots()

    def write_frame() -> None:
        frame = _draw_frame(ax, positions, edges)
        writer.append_data(frame)

    write_frame()
    with open(delta_log_path, "r", encoding="utf-8") as fh:
        for idx, line in enumerate(fh):
            if max_frames is not None and idx >= max_frames:
                break
            delta = json.loads(line)
            for nid, pos in delta.get("node_positions", {}).items():
                positions[int(nid)] = tuple(pos)
            edges.extend(tuple(e) for e in delta.get("edges", []))
            write_frame()

    writer.close()
    plt.close(fig)


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render snapshot logs to MP4")
    parser.add_argument("graph_static", help="Path to GraphStatic JSON")
    parser.add_argument("delta_log", help="Path to SnapshotDelta log (jsonl)")
    parser.add_argument("output", help="Output MP4 file")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    parser.add_argument(
        "--max-frames", type=int, default=None, help="Optional frame cap"
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> None:
    args = _parse_args(argv)
    render_video(
        args.graph_static, args.delta_log, args.output, args.fps, args.max_frames
    )


if __name__ == "__main__":  # pragma: no cover - CLI convenience
    main()
