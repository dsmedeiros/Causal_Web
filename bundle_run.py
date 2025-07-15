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
    "interference_log.json",
    "tick_density_map.json",
    "refraction_log.json",
    "node_state_log.json",
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

    # interference metrics
    inter_lines = _load_lines(os.path.join(out_dir, "interference_log.json"))
    max_inter = 0
    for entry in inter_lines:
        tick, states = next(iter(entry.items()))
        if states:
            m = max(states.values())
            if m > max_inter:
                max_inter = m
    manifest["max_interference_density"] = max_inter

    # decoherence zone width
    deco_lines_dict = {int(list(d.keys())[0]): list(d.values())[0] for d in deco_lines}
    widths = []
    for node in {n for entry in deco_lines for n in list(entry.values())[0].keys()}:
        streak = 0
        prev_tick = None
        for t in sorted(deco_lines_dict.keys()):
            val = deco_lines_dict[t].get(node)
            if val is not None and val > 0.4:
                if prev_tick is None or t == prev_tick + 1:
                    streak += 1
                else:
                    if streak:
                        widths.append(streak)
                    streak = 1
                prev_tick = t
        if streak:
            widths.append(streak)
    manifest["mean_decoherence_zone_width"] = round(sum(widths) / len(widths), 2) if widths else 0

    # phase drift range
    law_lines = _load_lines(os.path.join(out_dir, "law_wave_log.json"))
    freqs = []
    for entry in law_lines:
        _, vals = next(iter(entry.items()))
        freqs.extend(vals.values())
    manifest["phase_drift_range"] = round(max(freqs) - min(freqs), 4) if freqs else 0

    # collapsed nodes total
    if collapse_lines:
        last_states = list(collapse_lines)[-1]
        _, states = next(iter(last_states.items()))
        manifest["collapsed_nodes_total"] = sum(1 for s in states.values() if s)
    else:
        manifest["collapsed_nodes_total"] = 0

    # coherence stabilizers
    node_state_lines = _load_lines(os.path.join(out_dir, "node_state_log.json"))
    stabilizers = set()
    for entry in node_state_lines:
        _, info = next(iter(entry.items()))
        credit = info.get("credit", {})
        debt = info.get("debt", {})
        for node, c in credit.items():
            if c > debt.get(node, 0):
                stabilizers.add(node)
    manifest["coherence_stabilizers_count"] = len(stabilizers)

    # refraction events
    manifest["refraction_events_logged"] = len(_load_lines(os.path.join(out_dir, "refraction_log.json")))

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
