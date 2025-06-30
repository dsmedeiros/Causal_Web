import time
import threading
from config import Config
from .graph import CausalGraph
import json

graph = CausalGraph()

def build_graph():
    graph.add_node("A1", x=100, y=100, frequency=1.0)
    graph.add_node("A2", x=100, y=200, frequency=1.2)
    graph.add_node("C", x=300, y=150, frequency=1.0)

    graph.add_edge("A1", "C", delay=5)
    graph.add_edge("A2", "C", delay=5)

    # Both upstream nodes tick deterministically
    # for t in range(0, 50):
    #     graph.get_node("A1").tick_history.append((t, graph.get_node("A1").compute_phase(t)))
    #     graph.get_node("A2").tick_history.append((t, graph.get_node("A2").compute_phase(t)))

def simulation_loop():
    def run():
        global_tick = 0
        while Config.is_running:
            print(f"== Tick {global_tick} ==")

            # Propagate all past ticks forward
            for edge in graph.edges:
                source_node = graph.get_node(edge.source)
                target_node = graph.get_node(edge.target)

                for (tick_time, phase) in source_node.tick_history:
                    if tick_time + edge.delay == global_tick:
                        target_node.schedule_tick(global_tick, phase)

            # Try to tick target nodes
            for node_id, node in graph.nodes.items():
                node.maybe_tick(global_tick)

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