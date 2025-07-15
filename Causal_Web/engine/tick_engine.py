import time
import threading
from config import Config
from .graph import CausalGraph
from .observer import Observer
from .log_interpreter import run_interpreter
from .tick_seeder import TickSeeder
import json
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor
import math

# Global graph instance
graph = CausalGraph()
observers = []
kappa = 0.5  # curvature strength for refraction fields
_law_wave_stability = {}
seeder = TickSeeder(graph)

# Phase 6 metrics
void_absorption_events = 0
boundary_interactions_count = 0
bridges_reformed_count = 0
_decay_durations = []


def apply_global_forcing(tick: int) -> None:
    """Apply rhythmic modulation to all nodes."""
    jitter = Config.phase_jitter
    wave = Config.coherence_wave
    for node in graph.nodes.values():
        if jitter["amplitude"] and jitter.get("period", 0):
            node.phase += jitter["amplitude"] * math.sin(2 * math.pi * tick / jitter["period"])
        if wave["amplitude"] and wave.get("period", 0):
            mod = wave["amplitude"] * math.sin(2 * math.pi * tick / wave["period"])
            node.current_threshold = max(0.1, node.current_threshold - mod)


def clear_output_directory():
    """Remove or truncate JSON output files from previous runs."""
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "output"))
    if not os.path.isdir(out_dir):
        return
    for name in os.listdir(out_dir):
        if name == "__init__.py":
            continue
        open(os.path.join(out_dir, name), "w").close()

def build_graph():
    clear_output_directory()
    graph.load_from_file("input/graph.json")
    global seeder
    seeder = TickSeeder(graph)

def add_observer(observer: Observer):
    observers.append(observer)

def emit_ticks(global_tick):
    """Seed ticks into the graph via the configured seeder."""
    seeder.seed(global_tick)

def propagate_phases(global_tick):
    """Propagate phases scheduled during node ticks.

    Previous versions walked through each node's ``tick_history`` and
    rescheduled downstream ticks on every global step. This caused phases to
    arrive after twice the intended edge delay. Node.apply_tick now schedules
    outgoing phases directly, so this function no longer performs any work but
    is kept for API compatibility.
    """
    pass

def evaluate_nodes(global_tick):
    for node in graph.nodes.values():
        node.maybe_tick(global_tick, graph)

def log_curvature_per_tick(global_tick):
    log = {}
    for edge in graph.edges:
        src = graph.get_node(edge.source)
        tgt = graph.get_node(edge.target)
        if not src or not tgt:
            continue
        df = abs(src.law_wave_frequency - tgt.law_wave_frequency)
        curved = edge.adjusted_delay(src.law_wave_frequency, tgt.law_wave_frequency, kappa)
        log[f"{edge.source}->{edge.target}"] = {"delta_f": round(df,4), "curved_delay": round(curved,4)}
    with open("output/curvature_log.json", "a") as f:
        f.write(json.dumps({str(global_tick): log}) + "\n")

def log_bridge_states(global_tick):
    snapshot = {
        b.bridge_id: {
            "active": b.active,
            "last_activation": b.last_activation,
            "last_rupture_tick": b.last_rupture_tick,
            "last_reform_tick": b.last_reform_tick,
            "coherence_at_reform": b.coherence_at_reform,
            "trust_score": b.trust_score,
            "reinforcement": b.reinforcement_streak
        }
        for b in graph.bridges
    }
    with open("output/bridge_state_log.json", "a") as f:
        f.write(json.dumps({str(global_tick): snapshot}) + "\n")


def log_meta_node_ticks(global_tick):
    events = {}
    for meta_id, meta in graph.meta_nodes.items():
        member_ticks = [nid for nid in meta.member_ids
                        if any(t.time == global_tick for t in graph.get_node(nid).tick_history)]
        if member_ticks:
            events[meta_id] = member_ticks
        if events:
            with open("output/meta_node_tick_log.json", "a") as f:
                f.write(json.dumps({str(global_tick): events}) + "\n")


def snapshot_graph(global_tick):
    interval = getattr(Config, "snapshot_interval", 0)
    if interval and global_tick % interval == 0:
        path_dir = os.path.join("output", "runtime_graph_snapshots")
        os.makedirs(path_dir, exist_ok=True)
        path = os.path.join(path_dir, f"graph_{global_tick}.json")
        with open(path, "w") as f:
            json.dump(graph.to_dict(), f, indent=2)


def dynamic_bridge_management(global_tick):
    ids = list(graph.nodes.keys())
    existing = {(b.node_a_id, b.node_b_id) for b in graph.bridges}
    existing |= {(b.node_b_id, b.node_a_id) for b in graph.bridges}
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a = graph.get_node(ids[i])
            b = graph.get_node(ids[j])
            if (a.id, b.id) in existing:
                continue
            drift = abs((a.phase - b.phase + math.pi) % (2 * math.pi) - math.pi)
            if drift < 0.1 and a.coherence > 0.9 and b.coherence > 0.9:
                graph.add_bridge(
                    a.id,
                    b.id,
                    formed_at_tick=global_tick,
                    seeded=False,
                )
                bridge = graph.bridges[-1]
                bridge._log_dynamics(
                    global_tick,
                    "formed",
                    {
                        "phase_delta": drift,
                        "coherence": (a.coherence + b.coherence) / 2,
                    },
                )
                bridge.update_state(global_tick)


def _compute_metrics(node, tick_time):
    decoherence = node.compute_decoherence_field(tick_time)
    coherence = node.compute_coherence_level(tick_time)
    interference = len(node.pending_superpositions.get(tick_time, []))
    return node.id, decoherence, coherence, interference, node.node_type.value, node.coherence_credit, node.decoherence_debt


def log_metrics_per_tick(global_tick):
    decoherence_log = {}
    coherence_log = {}
    classical_state = {}
    coherence_velocity = {}
    law_wave_log = {}
    interference_log = {}
    credit_log = {}
    debt_log = {}
    type_log = {}

    # Store last coherence to compute delta
    if not hasattr(log_metrics_per_tick, "_last_coherence"):
        log_metrics_per_tick._last_coherence = {}

    with ThreadPoolExecutor() as ex:
        results = list(ex.map(lambda n: _compute_metrics(n, global_tick), graph.nodes.values()))

    for node_id, decoherence, coherence, interference, ntype, credit, debt in results:
        node = graph.get_node(node_id)
        prev = log_metrics_per_tick._last_coherence.get(node_id, coherence)
        delta = coherence - prev
        log_metrics_per_tick._last_coherence[node_id] = coherence
        node.coherence_velocity = delta

        node.update_classical_state(decoherence, tick_time=global_tick, graph=graph)

        # track law-wave stability
        record = _law_wave_stability.setdefault(node_id, {"freqs": [], "stable": 0})
        record["freqs"].append(node.law_wave_frequency)
        if len(record["freqs"]) > 5:
            record["freqs"].pop(0)
        if len(record["freqs"]) == 5:
            if np.std(record["freqs"]) < 0.01:
                record["stable"] += 1
            else:
                record["stable"] = 0
        if record["stable"] >= 10:
            node.refractory_period = max(1.0, node.refractory_period - 0.1)
            with open("output/law_drift_log.json", "a") as f:
                f.write(json.dumps({"tick": global_tick, "node": node_id, "new_refractory_period": node.refractory_period}) + "\n")
            record["stable"] = 0

        decoherence_log[node_id] = round(decoherence, 4)
        coherence_log[node_id] = round(coherence, 4)
        classical_state[node_id] = getattr(node, "is_classical", False)
        coherence_velocity[node_id] = round(delta, 5)
        law_wave_log[node_id] = round(node.law_wave_frequency, 4)
        interference_log[node_id] = interference
        credit_log[node_id] = round(credit, 3)
        debt_log[node_id] = round(debt, 3)
        type_log[node_id] = ntype

    clusters = graph.detect_clusters()
    graph.create_meta_nodes(clusters)

    with open("output/cluster_log.json", "a") as f:
        f.write(json.dumps({str(global_tick): clusters}) + "\n")

    with open("output/law_wave_log.json", "a") as f:
        f.write(json.dumps({str(global_tick): law_wave_log}) + "\n")

    with open("output/decoherence_log.json", "a") as f:
        f.write(json.dumps({str(global_tick): decoherence_log}) + "\n")
    with open("output/coherence_log.json", "a") as f:
        f.write(json.dumps({str(global_tick): coherence_log}) + "\n")
    with open("output/coherence_velocity_log.json", "a") as f:
        f.write(json.dumps({str(global_tick): coherence_velocity}) + "\\n")
    with open("output/classicalization_map.json", "a") as f:
        f.write(json.dumps({str(global_tick): classical_state}) + "\n")
    with open("output/interference_log.json", "a") as f:
        f.write(json.dumps({str(global_tick): interference_log}) + "\n")
    with open("output/tick_density_map.json", "a") as f:
        f.write(json.dumps({str(global_tick): interference_log}) + "\n")
    with open("output/node_state_log.json", "a") as f:
        f.write(json.dumps({str(global_tick): {"type": type_log, "credit": credit_log, "debt": debt_log}}) + "\n")


def simulation_loop():
    def run():
        global_tick = 0
        while Config.is_running:
            print(f"== Tick {global_tick} ==")

            apply_global_forcing(global_tick)

            emit_ticks(global_tick)
            propagate_phases(global_tick)
            evaluate_nodes(global_tick)

            log_metrics_per_tick(global_tick)
            log_bridge_states(global_tick)
            log_meta_node_ticks(global_tick)
            log_curvature_per_tick(global_tick)
            dynamic_bridge_management(global_tick)
            snapshot_graph(global_tick)

            for obs in observers:
                obs.observe(graph, global_tick)
                inferred = obs.infer_field_state()
                with open("output/observer_perceived_field.json", "a") as f:
                    f.write(json.dumps({"tick": global_tick, "observer": obs.id, "state": inferred}) + "\n")

                actual = {n.id: len(n.tick_history) for n in graph.nodes.values()}
                diff = {
                    nid: {"actual": actual.get(nid, 0), "inferred": inferred.get(nid, 0)}
                    for nid in set(actual) | set(inferred)
                    if actual.get(nid, 0) != inferred.get(nid, 0)
                }
                if diff:
                    with open("output/observer_disagreement_log.json", "a") as f:
                        f.write(json.dumps({"tick": global_tick, "observer": obs.id, "diff": diff}) + "\n")

            for bridge in graph.bridges:
                bridge.apply(global_tick, graph)

            Config.current_tick = global_tick

            if Config.max_ticks and global_tick >= Config.max_ticks:
                Config.is_running = False
                write_output()

            global_tick += 1
            time.sleep(Config.tick_rate)

    threading.Thread(target=run, daemon=True).start()

def write_output():
    with open("output/tick_trace.json", "w") as f:
        json.dump(graph.to_dict(), f, indent=2)
    print("✅ Tick trace saved to output/tick_trace.json")

    inspection = graph.inspect_superpositions()
    with open("output/inspection_log.json", "w") as f:
        json.dump(inspection, f, indent=2)
    print("✅ Superposition inspection saved to output/inspection_log.json")

    export_curvature_map()
    export_regional_maps()
    export_global_diagnostics()
    run_interpreter()

def export_curvature_map():
    """Aggregate curvature logs into a D3-friendly dataset."""
    grid = []
    try:
        with open("output/curvature_log.json") as f:
            for line in f:
                data = json.loads(line.strip())
                tick, edges = next(iter(data.items()))
                records = [
                    {
                        "source": k.split("->")[0],
                        "target": k.split("->")[1],
                        "delay": v["curved_delay"],
                    }
                    for k, v in edges.items()
                ]
                grid.append({"tick": int(tick), "edges": records})
    except FileNotFoundError:
        return

    with open("output/curvature_map.json", "w") as f:
        json.dump(grid, f, indent=2)
    print("✅ Curvature map exported to output/curvature_map.json")


def export_global_diagnostics():
    """Simple run-level metrics for diagnostics."""
    deco_lines = []
    try:
        with open("output/decoherence_log.json") as f:
            for line in f:
                deco_lines.append(json.loads(line))
    except FileNotFoundError:
        pass
    if not deco_lines:
        return
    first = next(iter(deco_lines[0].values()))
    last = next(iter(deco_lines[-1].values()))
    entropy_first = sum(first.values())
    entropy_last = sum(last.values())
    entropy_delta = entropy_last - entropy_first

    stability = []
    for entry in deco_lines:
        vals = list(entry.values())[0]
        stability.append(sum(1 for v in vals.values() if v < 0.4) / len(vals))
    coherence_stability_score = round(sum(stability) / len(stability), 3)

    collapse_events = 0
    try:
        with open("output/classicalization_map.json") as f:
            for line in f:
                states = json.loads(line)
                collapse_events += sum(1 for v in next(iter(states.values())).values() if v)
    except FileNotFoundError:
        pass
    resilience = 0
    try:
        with open("output/law_wave_log.json") as f:
            resilience = sum(1 for _ in f)
    except FileNotFoundError:
        pass
    collapse_resilience_index = round(resilience / (collapse_events or 1), 3)

    adaptivity = 0
    if graph.bridges:
        active = sum(1 for b in graph.bridges if b.active)
        adaptivity = active / len(graph.bridges)

    diagnostics = {
        "coherence_stability_score": coherence_stability_score,
        "entropy_delta": round(entropy_delta, 3),
        "collapse_resilience_index": collapse_resilience_index,
        "network_adaptivity_index": round(adaptivity, 3),
    }
    with open("output/global_diagnostics.json", "w") as f:
        json.dump(diagnostics, f, indent=2)
    print("✅ Global diagnostics exported")


def export_regional_maps():
    regions = {}
    for nid, node in graph.nodes.items():
        region = nid[0]
        regions.setdefault(region, []).append(node)

    regional_pressure = {}
    for reg, nodes in regions.items():
        pressure = sum(n.collapse_pressure for n in nodes) / len(nodes)
        regional_pressure[reg] = round(pressure, 3)

    matrix = {}
    for edge in graph.edges:
        src_r = edge.source[0]
        tgt_r = edge.target[0]
        key = f"{src_r}->{tgt_r}"
        matrix[key] = matrix.get(key, 0) + 1

    with open("output/regional_pressure_map.json", "w") as f:
        json.dump(regional_pressure, f, indent=2)
    with open("output/cluster_influence_matrix.json", "w") as f:
        json.dump(matrix, f, indent=2)
    print("✅ Regional influence maps exported")
