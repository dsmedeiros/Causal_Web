import time
import threading
from config import Config
from .graph import CausalGraph
import json

# Global graph instance
graph = CausalGraph()

def build_graph():
    graph.load_from_file("input/graph.json")

def emit_ticks(global_tick):
    for node in graph.nodes.values():
        node.emit_tick_if_ready(global_tick, graph)

def propagate_phases(global_tick):
    for edge in graph.edges:
        source_node = graph.get_node(edge.source)
        for tick_time, phase in source_node.tick_history:
            if tick_time + edge.adjusted_delay() == global_tick:
                edge.propagate_phase(phase, global_tick, graph)

def evaluate_nodes(global_tick):
    for node in graph.nodes.values():
        node.maybe_tick(global_tick, graph)

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

    # Store last coherence to compute delta
    if not hasattr(log_metrics_per_tick, "_last_coherence"):
        log_metrics_per_tick._last_coherence = {}

    for node_id, node in graph.nodes.items():
        decoherence = node.compute_decoherence_field(global_tick)
        coherence = node.compute_coherence_level(global_tick)
        prev = log_metrics_per_tick._last_coherence.get(node_id, coherence)
        delta = coherence - prev
        log_metrics_per_tick._last_coherence[node_id] = coherence

        node.update_classical_state(decoherence)

        decoherence_log[node_id] = round(decoherence, 4)
        coherence_log[node_id] = round(coherence, 4)
        classical_state[node_id] = getattr(node, "is_classical", False)
        coherence_velocity[node_id] = round(delta, 5)

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
    print("✅ Superposition inspection saved to output/superposition_inspection.json")
