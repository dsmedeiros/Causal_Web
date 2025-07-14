import dearpygui.dearpygui as dpg
from config import Config
from engine.tick_engine import simulation_loop
from engine.tick_engine import graph, build_graph
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

def pause_resume_callback():
    Config.is_running = not Config.is_running
    if dpg.does_item_exist("pause_button"):
        dpg.set_value("pause_button", "Pause" if Config.is_running else "Resume")

def tick_rate_changed(sender, app_data):
    Config.tick_rate = app_data

def max_ticks_changed(sender, app_data):
    Config.max_ticks = app_data

def start_sim_callback():
    if not Config.is_running:
        Config.is_running = True
        simulation_loop()
        dpg.configure_item("start_button", enabled=False)

def update_display():
    if dpg.does_item_exist("tick_counter"):
        dpg.set_value("tick_counter", f"Tick: {Config.current_tick}")

def update_graph_visuals():
    # for node_id in graph.nodes:
    #     tick_count = len(graph.get_node(node_id).tick_history)
    #     label_tag = f"{node_id}_label"
    #     if dpg.does_item_exist(label_tag):
    #         dpg.set_value(label_tag, f"{node_id}\\nTicks: {tick_count}")
    #         print(f"[GUI] {node_id} ticks: {tick_count}")
    if not dpg.does_item_exist("graph_drawlist"):
        return

    dpg.delete_item("graph_drawlist", children_only=True)

    cluster_map = {}
    clusters = graph.detect_clusters()
    for idx, c in enumerate(clusters, start=1):
        for nid in c:
            cluster_map[nid] = idx

    for node_id, node in graph.nodes.items():
        x, y = node.x, node.y
        tick_count = len(node.tick_history)
        vel = getattr(node, "coherence_velocity", 0.0)

        color = (100, 149, 237) if "A1" in node_id else (60, 179, 113) if "A2" in node_id else (220, 20, 60)

        dpg.draw_circle(center=(x, y), radius=30, color=color, fill=color, parent="graph_drawlist")

        label = f"{node_id}\nTicks: {tick_count}\nVel:{vel:.2f}"
        if node_id in cluster_map:
            label += f"\nC{cluster_map[node_id]}"
        dpg.draw_text(pos=(x - 30, y - 15), text=label, size=15, color=(255, 255, 255), parent="graph_drawlist")

    # Draw edges with curvature overlays
    for edge in graph.edges:
        from_node = graph.nodes[edge.source]
        to_node = graph.nodes[edge.target]

        delay = edge.adjusted_delay(from_node.law_wave_frequency, to_node.law_wave_frequency)
        thickness = 2 + delay * 0.2
        intensity = min(255, int(50 + delay * 20))
        color = (200, 200 - intensity if 200 - intensity > 0 else 0, 200)

        dpg.draw_arrow(p1=(from_node.x, from_node.y),
                       p2=(to_node.x, to_node.y),
                       color=color,
                       thickness=thickness,
                       parent="graph_drawlist")


def refresh():
    update_graph_visuals()
    if dpg.does_item_exist("tick_counter"):
        dpg.set_value("tick_counter", f"Tick: {Config.current_tick}")
    dpg.render_dearpygui_frame()

def launch():
    dashboard()

def gui_update_callback():
    update_graph_visuals()
    if dpg.does_item_exist("tick_counter"):
        dpg.set_value("tick_counter", f"Tick: {Config.current_tick}")
    next_frame = dpg.get_frame_count() + 1
    dpg.set_frame_callback(next_frame, gui_update_callback)

def dashboard():
    build_graph()
    dpg.create_context()

    with dpg.font_registry():
        default_font = dpg.add_font("assets/fonts/consola.ttf", 20)

    dpg.bind_font(default_font)
    dpg.create_viewport(title="CWT Simulation Dashboard", width=800, height=800)

    with dpg.window(label="Control Panel", width=800, height=200):
        dpg.add_slider_float(label="Tick Rate (sec)", default_value=Config.tick_rate,
                             min_value=0.1, max_value=2.0, callback=tick_rate_changed, tag="tick_rate_slider")
        dpg.add_input_int(label="Max Ticks", default_value=Config.max_ticks,
                          callback=max_ticks_changed, tag="max_ticks_input")
        dpg.add_button(label="Pause", callback=pause_resume_callback, tag="pause_button")
        dpg.add_button(label="Start Simulation", callback=start_sim_callback, tag="start_button")
        dpg.add_text("Tick: 0", tag="tick_counter")

    with dpg.window(label="Causal Graph", width=800, height=460):
        with dpg.drawlist(width=600, height=400, tag="graph_drawlist"):
            for node_id, (x, y) in node_positions.items():
                color = node_colors[node_id]
                node_tag = dpg.draw_circle(center=(x, y), radius=30, color=color, fill=color)
                label_tag = dpg.draw_text((x - 25, y - 10), f"{node_id}\\nTicks: 0", size=15, tag=f"{node_id}_label")
                node_tags[node_id] = node_tag
                tick_counters[node_id] = label_tag

            # Draw directed edges
            dpg.draw_arrow(p1=node_positions["A1"], p2=node_positions["C"], color=(150, 150, 150), thickness=2)
            dpg.draw_arrow(p1=node_positions["A2"], p2=node_positions["C"], color=(150, 150, 150), thickness=2)

    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_frame_callback(1, gui_update_callback)
    dpg.start_dearpygui()
    dpg.destroy_context()

