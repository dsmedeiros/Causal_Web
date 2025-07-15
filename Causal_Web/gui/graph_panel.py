import os
import dearpygui.dearpygui as dpg
from ..config import Config
from ..engine.tick_engine import simulation_loop
from ..engine.tick_engine import graph, recent_csp_failures
import threading

# Tick stores for visual update
node_colors = {
    "A1": (100, 149, 237),  # Cornflower Blue
    "A2": (60, 179, 113),  # Medium Sea Green
    "C": (220, 20, 60),  # Crimson
}

# Mapping of node origin types to display colors
ORIGIN_COLORS = {
    "SIP_BUD": node_colors["A2"],
    "SIP_RECOMB": node_colors["A1"],
    "CSP": node_colors["C"],
}
node_positions = {"A1": (100, 200), "A2": (100, 300), "C": (400, 250)}
node_tags = {}
tick_counters = {}


def update_graph_visuals():
    """Redraw the graph nodes, edges and CSP failure ripples."""
    if not dpg.does_item_exist("graph_drawlist"):
        return

    dpg.delete_item("graph_drawlist", children_only=True)

    for node_id, node in graph.nodes.items():
        color = ORIGIN_COLORS.get(
            getattr(node, "origin_type", ""), node_colors.get(node_id, (100, 149, 237))
        )
        dpg.draw_circle(
            center=(node.x, node.y),
            radius=30,
            color=color,
            fill=color,
            parent="graph_drawlist",
        )

        label = f"{node_id}\nTicks: {len(node.tick_history)}"
        dpg.draw_text(
            pos=(node.x - 25, node.y - 10),
            text=label,
            size=15,
            parent="graph_drawlist",
        )

    for edge in graph.edges:
        from_node = graph.nodes[edge.source]
        to_node = graph.nodes[edge.target]
        delay = edge.adjusted_delay(
            from_node.law_wave_frequency, to_node.law_wave_frequency
        )
        thickness = 2 + delay * 0.2
        intensity = min(255, int(50 + delay * 20))
        color = (150, 150 - intensity if 150 - intensity > 0 else 0, 150)
        dpg.draw_arrow(
            p1=(from_node.x, from_node.y),
            p2=(to_node.x, to_node.y),
            color=color,
            thickness=thickness,
            parent="graph_drawlist",
        )

    now = Config.current_tick
    for rip in list(recent_csp_failures):
        age = now - rip["tick"]
        if age > 10:
            recent_csp_failures.remove(rip)
            continue
        radius = 10 + age * 7
        alpha = max(0, 255 - age * 25)
        dpg.draw_circle(
            center=(rip["x"], rip["y"]),
            radius=radius,
            color=(255, 128, 128, alpha),
            fill=(255, 128, 128, 50),
            parent="graph_drawlist",
        )


def gui_loop():
    dpg.create_context()
    dpg.create_viewport(title="CWT Real-Time Causal Graph", width=640, height=480)

    with dpg.font_registry():
        font_path = os.path.join(
            os.path.dirname(__file__), "..", "assets", "fonts", "consola.ttf"
        )
        default_font = dpg.add_font(font_path, 18)
    dpg.bind_font(default_font)

    with dpg.window(label="Causal Graph", width=620, height=460):
        with dpg.drawlist(width=600, height=400, tag="graph_drawlist"):
            pass
        dpg.add_button(label="Start Simulation", callback=start_sim)
        dpg.add_text("Tick: 0", tag="tick_counter")

    with dpg.window(label="Legend", width=180, height=100, pos=(450, 10)):
        dpg.add_text(
            "\u2b1b SIP_BUD — Stable propagation", color=ORIGIN_COLORS["SIP_BUD"]
        )
        dpg.add_text(
            "\u2b1b SIP_RECOMB — Recombination", color=ORIGIN_COLORS["SIP_RECOMB"]
        )
        dpg.add_text("\u2b1b CSP — Collapse-seeded", color=ORIGIN_COLORS["CSP"])

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
