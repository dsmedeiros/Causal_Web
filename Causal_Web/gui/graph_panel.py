import dearpygui.dearpygui as dpg
from ..config import Config
from ..engine.tick_engine import simulation_loop
from ..engine.tick_engine import graph
import threading

# Tick stores for visual update
node_colors = {
    "A1": (100, 149, 237),  # Cornflower Blue
    "A2": (60, 179, 113),   # Medium Sea Green
    "C":  (220, 20, 60)     # Crimson
}
node_positions = {
    "A1": (100, 200),
    "A2": (100, 300),
    "C":  (400, 250)
}
node_tags = {}
tick_counters = {}

def update_graph_visuals():
    for node_id in graph.nodes:
        tick_count = len(graph.get_node(node_id).tick_history)
        dpg.set_value(f"{node_id}_label", f"{node_id}\\nTicks: {tick_count}")

def gui_loop():
    dpg.create_context()
    dpg.create_viewport(title="CWT Real-Time Causal Graph", width=640, height=480)

    with dpg.font_registry():
        default_font = dpg.add_font("assets/fonts/consola.ttf", 18)
    dpg.bind_font(default_font)

    with dpg.window(label="Causal Graph", width=620, height=460):
        with dpg.drawlist(width=600, height=400):
            for node_id, (x, y) in node_positions.items():
                color = node_colors[node_id]
                node_tag = dpg.draw_circle(center=(x, y), radius=30, color=color, fill=color)
                label_tag = dpg.draw_text((x - 25, y - 10), f"{node_id}\\nTicks: 0", size=15)
                node_tags[node_id] = node_tag
                tick_counters[node_id] = label_tag

            # Draw directed edges with curvature overlays
            for edge in graph.edges:
                from_node = graph.nodes[edge.source]
                to_node = graph.nodes[edge.target]
                delay = edge.adjusted_delay(from_node.law_wave_frequency, to_node.law_wave_frequency)
                thickness = 2 + delay * 0.2
                intensity = min(255, int(50 + delay * 20))
                color = (150, 150 - intensity if 150 - intensity > 0 else 0, 150)
                dpg.draw_arrow(p1=(from_node.x, from_node.y), p2=(to_node.x, to_node.y), color=color, thickness=thickness)

        dpg.add_button(label="Start Simulation", callback=start_sim)
        dpg.add_text("Tick: 0", tag="tick_counter")

    dpg.setup_dearpygui()
    dpg.show_viewport()

    def refresh():
        while dpg.is_dearpygui_running():
            update_graph_visuals()
            dpg.set_value("tick_counter", f"Tick: {Config.current_tick}")
            dpg.render_dearpygui_frame()

    threading.Thread(target=refresh, daemon=True).start()
    dpg.start_dearpygui()
    dpg.destroy_context()

def start_sim():
    if not Config.is_running:
        Config.is_running = True
        simulation_loop()
