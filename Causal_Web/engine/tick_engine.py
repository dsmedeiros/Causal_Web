import time
import threading
from config import Config
from .graph import CausalGraph
from .observer import Observer
import json
import numpy as np

# Global graph instance
graph = CausalGraph()
observers = []
kappa = 0.5  # curvature strength for refraction fields
_law_wave_stability = {}

def build_graph():
    graph.load_from_file("input/graph.json")

def add_observer(observer: Observer):
    observers.append(observer)

def emit_ticks(global_tick):
    for source in getattr(graph, "tick_sources", []):
        node = graph.get_node(source["node_id"])
        interval = source.get("tick_interval", 1)
        phase = source.get("phase", 0.0)
        if node and not node.is_classical and global_tick % interval == 0:
            node.apply_tick(global_tick, phase, graph, origin="source")

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
            "coherence_at_reform": b.coherence_at_reform
        }
        for b in graph.bridges
    }
    with open("output/bridge_state_log.json", "a") as f:
        f.write(json.dumps({str(global_tick): snapshot}) + "\n")


def log_metrics_per_tick(global_tick):
    decoherence_log = {}
    coherence_log = {}
    classical_state = {}
    coherence_velocity = {}
    law_wave_log = {}

    # Store last coherence to compute delta
    if not hasattr(log_metrics_per_tick, "_last_coherence"):
        log_metrics_per_tick._last_coherence = {}

    for node_id, node in graph.nodes.items():
        decoherence = node.compute_decoherence_field(global_tick)
        coherence = node.compute_coherence_level(global_tick)
        prev = log_metrics_per_tick._last_coherence.get(node_id, coherence)
        delta = coherence - prev
        log_metrics_per_tick._last_coherence[node_id] = coherence
        node.coherence_velocity = delta

        node.update_classical_state(decoherence, tick_time=global_tick)

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

    clusters = graph.detect_clusters()
    graph.create_meta_nodes(clusters)

    with open("output/cluster_log.json", "a") as f:
        f.write(json.dumps({str(global_tick): clusters}) + "\n")

    with open("output/law_wave_log.json", "a") as f:
        f.write(json.dumps({str(global_tick): law_wave_log}) + "\n")

    clusters = graph.detect_clusters()

    with open("output/cluster_log.json", "a") as f:
        f.write(json.dumps({str(global_tick): clusters}) + "\n")

    with open("output/decoherence_log.json", "a") as f:
        f.write(json.dumps({str(global_tick): decoherence_log}) + "\n")
    with open("output/coherence_log.json", "a") as f:
        f.write(json.dumps({str(global_tick): coherence_log}) + "\n")
    with open("output/coherence_velocity_log.json", "a") as f:
        f.write(json.dumps({str(global_tick): coherence_velocity}) + "\\n")
    with open("output/classicalization_map.json", "a") as f:
        f.write(json.dumps({str(global_tick): classical_state}) + "\n")


def simulation_loop():
    def run():
        global_tick = 0
        while Config.is_running:
            print(f"== Tick {global_tick} ==")

            emit_ticks(global_tick)
            propagate_phases(global_tick)
            evaluate_nodes(global_tick)

            log_metrics_per_tick(global_tick)
            log_bridge_states(global_tick)
            log_curvature_per_tick(global_tick)

            for obs in observers:
                obs.observe(graph, global_tick)
                inferred = obs.infer_field_state()
                with open("output/observer_perceived_field.json", "a") as f:
                    f.write(json.dumps({"tick": global_tick, "observer": obs.id, "state": inferred}) + "\n")

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
