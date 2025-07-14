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
        node.emit_tick_if_ready(global_tick)

def propagate_phases(global_tick):
    for edge in graph.edges:
        source_node = graph.get_node(edge.source)
        for tick_time, phase in source_node.tick_history:
            if tick_time + edge.adjusted_delay() == global_tick:
                edge.propagate_phase(phase, global_tick, graph)

def evaluate_nodes(global_tick):
    for node in graph.nodes.values():
        node.maybe_tick(global_tick, graph)

def simulation_loop():
    def run():
        global_tick = 0
        while Config.is_running:
            print(f"== Tick {global_tick} ==")

            emit_ticks(global_tick)
            propagate_phases(global_tick)
            evaluate_nodes(global_tick)

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
