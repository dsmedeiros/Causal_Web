"""Utility to extract simplified golden logs from engine runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def extract_golden(run_dir: str, out_path: str, *, max_frames: int = 500) -> None:
    """Write a trimmed golden log from ``run_dir`` to ``out_path``.

    ``run_dir`` is expected to contain a ``delta_log.jsonl`` file as produced
    by the engine. Only the ``frame`` index and ``residual_ewma`` invariant are
    retained to minimise repository size. ``max_frames`` limits the number of
    frames written to the golden log.
    """

    src = Path(run_dir) / "delta_log.jsonl"
    dest = Path(out_path)
    count = 0
    dest.parent.mkdir(parents=True, exist_ok=True)
    with src.open() as inp, dest.open("w") as out:
        for line in inp:
            rec = json.loads(line)
            frame = rec.get("frame")
            residual = rec.get("invariants", {}).get("residual_ewma")
            if residual is None:
                residual = rec.get("counters", {}).get("residual")
            json.dump({"frame": frame, "invariants": {"residual_ewma": residual}}, out)
            out.write("\n")
            count += 1
            if count >= max_frames:
                break


def main() -> None:
    """CLI entry point for extracting golden logs."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_dir", help="engine run directory containing delta_log.jsonl"
    )
    parser.add_argument("out", help="destination path for simplified golden log")
    parser.add_argument("--max-frames", type=int, default=500, dest="max_frames")
    args = parser.parse_args()
    extract_golden(args.run_dir, args.out, max_frames=args.max_frames)


if __name__ == "__main__":  # pragma: no cover - CLI tool
    main()
