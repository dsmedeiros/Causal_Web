"""CLI tool to bundle simulation outputs and run interpretation layers."""

import argparse
import json
import os
import shutil
import zipfile
from datetime import datetime, timezone

from Causal_Web.engine.logging.log_interpreter import run_interpreter
from Causal_Web.config import Config


DEFAULT_KEEP_FILES = [
    "tick_trace.json",
    "coherence_log.json",
    "decoherence_log.json",
    "bridge_state.json",
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
    "tick_seed_log.json",
    "tick_delivery_log.json",
    "refraction_log.json",
    "collapse_front_log.json",
    "collapse_chain_log.json",
    "layer_transition_log.json",
    "node_state_log.json",
    "node_state_map.json",
    "boundary_interaction_log.json",
    "bridge_reformation_log.json",
    "bridge_decay_log.json",
    "regional_pressure_map.json",
    "cluster_influence_matrix.json",
    "global_diagnostics.json",
    "void_node_map.json",
]


# ------------------------------------------------------------


def _create_manifest(out_dir: str, run_id: str, timestamp: str) -> None:
    """Generate manifest.json with run metadata."""
    manifest = {"run_id": run_id, "timestamp": timestamp}

    def _load_lines(path):
        """Return JSON lines from ``path`` or the unified log for that label."""
        if os.path.exists(path):
            with open(path) as f:
                return [json.loads(line) for line in f if line.strip()]

        name = os.path.basename(path)
        label = name.replace(".json", "")
        category = Config.category_for_file(name)
        if category == "event":
            base = os.path.join(out_dir, "events_log.jsonl")
            if not os.path.exists(base):
                return []
            result = []
            with open(base) as f:
                for line in f:
                    obj = json.loads(line)
                    if obj.get("event_type") == label:
                        result.append(obj)
            return result
        base = os.path.join(
            out_dir,
            "ticks_log.jsonl" if category == "tick" else "phenomena_log.jsonl",
        )
        if not os.path.exists(base):
            return []
        data = []
        with open(base) as f:
            for line in f:
                obj = json.loads(line)
                if obj.get("label") == label:
                    data.append({obj.get("tick"): obj.get("value")})
        return data

    deco_lines = _load_lines(os.path.join(out_dir, "decoherence_log.json"))
    manifest["total_ticks"] = len(deco_lines)

    seed_lines = _load_lines(os.path.join(out_dir, "tick_seed_log.json"))
    manifest["ticks_seeded"] = len(seed_lines)
    manifest["seeded_nodes_count"] = len({e.get("node") for e in seed_lines})
    manifest["seeding_strategy"] = Config.seeding.get("strategy", "static")

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
    meta_lines = _load_lines(os.path.join(out_dir, "meta_node_ticks.json"))
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
    manifest["bridge_ruptures"] = sum(
        1 for e in events if e.get("event_type") == "bridge_ruptured"
    )

    # law drift
    manifest["law_drift_events"] = len(
        _load_lines(os.path.join(out_dir, "law_drift_log.json"))
    )

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
    manifest["mean_decoherence_zone_width"] = (
        round(sum(widths) / len(widths), 2) if widths else 0
    )

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
    manifest["refraction_events_logged"] = len(
        _load_lines(os.path.join(out_dir, "refraction_log.json"))
    )

    # bridge dynamics
    dynamics = _load_lines(os.path.join(out_dir, "bridge_dynamics_log.json"))
    manifest["initial_bridges_count"] = manifest.get("bridge_count", 0)
    manifest["spontaneous_bridges_formed"] = sum(
        1 for e in dynamics if e.get("event") == "formed" and not e.get("seeded")
    )
    manifest["total_bridge_ruptures"] = sum(
        1 for e in dynamics if e.get("event") == "ruptured"
    )
    lifetimes = {}
    for e in dynamics:
        bid = e.get("bridge_id")
        if e.get("event") == "formed" and not e.get("seeded"):
            lifetimes[bid] = e.get("tick", 0)
        elif e.get("event") == "ruptured" and bid in lifetimes:
            start = lifetimes[bid]
            duration = e.get("tick", 0) - start
            lifetimes[bid] = duration
    completed = [v for v in lifetimes.values() if isinstance(v, int)]
    if completed:
        manifest["longest_lived_dynamic_bridge"] = max(completed)
        manifest["mean_bridge_lifetime"] = round(sum(completed) / len(completed), 2)
    else:
        manifest["longest_lived_dynamic_bridge"] = 0
        manifest["mean_bridge_lifetime"] = 0

    # law wave propagation
    law_wave_events = _load_lines(os.path.join(out_dir, "law_wave_event.json"))
    manifest["law_waves_emitted"] = sum(1 for e in law_wave_events if e.get("origin"))
    manifest["collapse_pressure_events"] = sum(
        len(e.get("affected", [])) for e in law_wave_events
    )

    # node state transitions
    transitions = _load_lines(os.path.join(out_dir, "node_state_map.json"))
    manifest["node_state_transitions_count"] = len(transitions)

    boundary_events = _load_lines(
        os.path.join(out_dir, "boundary_interaction_log.json")
    )
    manifest["boundary_interactions_count"] = len(boundary_events)
    manifest["void_absorption_events"] = sum(
        1 for e in boundary_events if e.get("void")
    )

    reforms = _load_lines(os.path.join(out_dir, "bridge_reformation_log.json"))
    manifest["bridges_reformed"] = len(reforms)

    decays = _load_lines(os.path.join(out_dir, "bridge_decay_log.json"))
    durations = [d.get("duration", 0) for d in decays]
    if durations:
        manifest["mean_decay_duration"] = round(sum(durations) / len(durations), 2)
    else:
        manifest["mean_decay_duration"] = 0

    try:
        with open(os.path.join(out_dir, "global_diagnostics.json")) as f:
            diag = json.load(f)
            manifest.update(diag)
    except FileNotFoundError:
        pass

    try:
        with open(os.path.join(out_dir, "regional_pressure_map.json")) as f:
            reg = json.load(f)
            manifest["regions_over_pressure_threshold"] = sum(
                1 for v in reg.values() if v > 1
            )
    except FileNotFoundError:
        manifest["regions_over_pressure_threshold"] = 0

    try:
        with open(os.path.join(out_dir, "cluster_influence_matrix.json")) as f:
            mat = json.load(f)
            manifest["max_interregional_flow"] = max(mat.values()) if mat else 0
    except FileNotFoundError:
        manifest["max_interregional_flow"] = 0

    try:
        collapse = _load_lines(os.path.join(out_dir, "collapse_chain_log.json"))
        manifest["collapse_clusters_detected"] = len(collapse)
    except Exception:
        manifest["collapse_clusters_detected"] = 0

    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)


# ------------------------------------------------------------


def bundle_run(output_dir: str, run_id: str, do_zip: bool = False) -> str:
    """Bundle logs and summaries for a simulation run.

    Parameters
    ----------
    output_dir:
        Directory containing simulation output files.
    run_id:
        Identifier incorporated into the bundled folder name.
    do_zip:
        When ``True`` compress the bundle into ``.cwbundle.zip``.

    Returns
    -------
    str
        Path to the bundled directory or archive.
    """

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M")
    out_dir = os.path.abspath(output_dir)

    # Run interpreter chain first
    run_interpreter(output_dir=out_dir)

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
    parser.add_argument(
        "--output-dir", default="Causal_Web/output", help="Simulation output directory"
    )
    parser.add_argument("--run-id", default="run", help="Run identifier")
    parser.add_argument(
        "--zip", action="store_true", help="Compress bundle into .cwbundle.zip"
    )
    args = parser.parse_args()

    path = bundle_run(args.output_dir, args.run_id, args.zip)
    print(f"✅ Bundle created at {path}")


if __name__ == "__main__":
    main()
