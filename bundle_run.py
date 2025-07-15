"""CLI tool to bundle simulation outputs and run interpretation layers."""

import argparse
import json
import os
import shutil
import zipfile
from datetime import datetime

from Causal_Web.engine.log_interpreter import run_interpreter


DEFAULT_KEEP_FILES = [
    "tick_trace.json",
    "coherence_log.json",
    "decoherence_log.json",
    "bridge_state_log.json",
    "cluster_log.json",
    "classicalization_map.json",
    "law_wave_log.json",
    "curvature_map.json",
    "interpretation_summary.txt",
    "interpretation_log.json",
    "causal_summary.txt",
    "causal_explanations.json",
    "explanation_graph.json",
    "manifest.json",
]


# ------------------------------------------------------------

def _create_manifest(out_dir: str, run_id: str, timestamp: str) -> None:
    """Generate manifest.json with run metadata."""
    manifest = {"run_id": run_id, "timestamp": timestamp}

    def _load_lines(path):
        if not os.path.exists(path):
            return []
        with open(path) as f:
            return [json.loads(line) for line in f if line.strip()]

    deco_lines = _load_lines(os.path.join(out_dir, "decoherence_log.json"))
    manifest["total_ticks"] = len(deco_lines)

    graph_path = os.path.join("Causal_Web", "input", "graph.json")
    if os.path.exists(graph_path):
        with open(graph_path) as f:
            graph = json.load(f)
        manifest["node_count"] = len(graph.get("nodes", []))
        manifest["bridge_count"] = len(graph.get("bridges", []))
    else:
        manifest["node_count"] = 0
        manifest["bridge_count"] = 0

    # meta nodes formed
    meta_lines = _load_lines(os.path.join(out_dir, "meta_node_tick_log.json"))
    metas = set()
    for entry in meta_lines:
        tick, events = next(iter(entry.items()))
        metas.update(events.keys())
    manifest["meta_nodes_formed"] = len(metas)

    # collapses
    collapse_lines = _load_lines(os.path.join(out_dir, "classicalization_map.json"))
    collapse_events = 0
    prev = {}
    for entry in collapse_lines:
        tick, states = next(iter(entry.items()))
        for node, state in states.items():
            if prev.get(node, False) != state and state:
                collapse_events += 1
            prev[node] = state
    manifest["collapse_events"] = collapse_events

    # bridge ruptures
    events = _load_lines(os.path.join(out_dir, "event_log.json"))
    manifest["bridge_ruptures"] = sum(1 for e in events if e.get("event_type") == "bridge_ruptured")

    # law drift
    manifest["law_drift_events"] = len(_load_lines(os.path.join(out_dir, "law_drift_log.json")))

    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)


# ------------------------------------------------------------

def bundle_run(output_dir: str, run_id: str, do_zip: bool = False) -> str:
    timestamp = datetime.utcnow().strftime("%Y-%m-%d_%H-%M")
    out_dir = os.path.abspath(output_dir)

    # Run interpreter chain first
    run_interpreter()

    _create_manifest(out_dir, run_id, timestamp)

    bundle_name = f"{run_id}_{timestamp}"
    dest = os.path.join(out_dir, bundle_name)
    os.makedirs(dest, exist_ok=True)

    for name in DEFAULT_KEEP_FILES:
        src = os.path.join(out_dir, name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(dest, name))

    graph_src = os.path.join("Causal_Web", "input", "graph.json")
    if os.path.exists(graph_src):
        shutil.copy2(graph_src, os.path.join(dest, "graph.json"))

    if do_zip:
        zip_path = f"{dest}.cwbundle.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, _, files in os.walk(dest):
                for f in files:
                    fp = os.path.join(root, f)
                    arcname = os.path.relpath(fp, dest)
                    zf.write(fp, arcname)
        return zip_path
    return dest


# ------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Bundle CWT run outputs")
    parser.add_argument("--output-dir", default="Causal_Web/output", help="Simulation output directory")
    parser.add_argument("--run-id", default="run", help="Run identifier")
    parser.add_argument("--zip", action="store_true", help="Compress bundle into .cwbundle.zip")
    args = parser.parse_args()

    path = bundle_run(args.output_dir, args.run_id, args.zip)
    print(f"✅ Bundle created at {path}")


if __name__ == "__main__":
    main()
