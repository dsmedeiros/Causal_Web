import os
import dearpygui.dearpygui as dpg
from .canvas import GraphCanvas
from .state import (
    get_active_file,
    get_selected_node,
    get_graph,
    set_graph,
    set_active_file,
    set_selected_node,
)
from . import connection_tool
from .toolbar import add_toolbar
from ..command_stack import AddNodeCommand
from .canvas import _commands
from ..graph.io import save_graph, load_graph
from ..config import Config
from ..engine.tick_engine import (
    simulation_loop,
    build_graph,
    _update_simulation_state,
)

# Tick stores for visual update
node_colors = {
    "A1": (100, 149, 237),
    "A2": (60, 179, 113),
    "C": (220, 20, 60),
}

# Mapping of origin types to colors for dynamic nodes
ORIGIN_COLORS = {
    "SIP_BUD": (60, 179, 113),  # green
    "SIP_RECOMB": (0, 255, 255),  # cyan
    "CSP": (220, 20, 60),  # crimson
}

ripples = []  # GUI animations for failed CSP seeds
node_positions = {"A1": (100, 200), "A2": (100, 300), "C": (400, 250)}
node_tags = {}
tick_counters = {}
canvas: GraphCanvas | None = None


def pause_resume_callback():
    with Config.state_lock:
        Config.is_running = not Config.is_running
        running = Config.is_running
        tick = Config.current_tick
    if dpg.does_item_exist("pause_button"):
        dpg.set_value("pause_button", "Pause" if running else "Resume")
    _update_simulation_state(not running, False, tick, None)


def tick_rate_changed(sender, app_data):
    with Config.state_lock:
        Config.tick_rate = app_data


def max_ticks_changed(sender, app_data):
    with Config.state_lock:
        Config.max_ticks = app_data


def max_children_changed(sender, app_data):
    with Config.state_lock:
        Config.max_children_per_node = app_data


def toggle_log(sender, app_data, user_data):
    """Enable or disable writing of a specific log file."""
    Config.log_files[user_data] = app_data


def _param_changed(sender, app_data, user_data):
    """Update :class:`Config` attributes when GUI inputs change."""
    parts = user_data.split(".")
    target = Config
    for part in parts[:-1]:
        target = getattr(target, part)
    if isinstance(target, dict):
        target[parts[-1]] = app_data
    else:
        setattr(target, parts[-1], app_data)


def _add_param_controls(data: dict, prefix: str = "") -> None:
    """Recursively generate input widgets for config parameters."""
    for key, value in data.items():
        full = f"{prefix}{key}"
        if isinstance(value, dict):
            dpg.add_text(full)
            with dpg.group():
                _add_param_controls(value, prefix=f"{full}.")
            continue
        if isinstance(value, bool):
            dpg.add_checkbox(
                label=full,
                default_value=value,
                callback=_param_changed,
                user_data=full,
            )
        elif isinstance(value, int):
            dpg.add_input_int(
                label=full,
                default_value=value,
                callback=_param_changed,
                user_data=full,
            )
        elif isinstance(value, float):
            dpg.add_input_float(
                label=full,
                default_value=value,
                callback=_param_changed,
                user_data=full,
            )
        else:
            dpg.add_input_text(
                label=full,
                default_value=str(value),
                callback=_param_changed,
                user_data=full,
            )


def add_node_callback() -> None:
    """Insert a new node into the editable graph."""
    model = get_graph()
    idx = 1
    while f"N{idx}" in model.nodes:
        idx += 1
    cmd = AddNodeCommand(model, f"N{idx}", {"x": 50.0 * idx, "y": 50.0 * idx})
    _commands.do(cmd)
    set_selected_node(f"N{idx}")
    if canvas is not None:
        canvas.build_full_canvas()


def start_sim_callback():
    path = get_active_file() or Config.input_path("graph.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_graph(path, get_graph())
    Config.new_run()
    build_graph()
    with Config.state_lock:
        if Config.is_running:
            return
        Config.is_running = True
        tick = Config.current_tick
    simulation_loop()
    dpg.configure_item("start_button", enabled=False)
    _update_simulation_state(False, False, tick, None)


def graph_resize_callback(sender, app_data, user_data):
    """Resize graph canvas to match its window."""
    width = dpg.get_item_width("graph_window") - 10
    height = dpg.get_item_height("graph_window") - 40
    child_width = dpg.get_item_width("graph_child")
    child_height = dpg.get_item_height("graph_child")
    if width != child_width or height != child_height:
        dpg.configure_item("graph_child", width=width, height=height)
        dpg.configure_item("graph_drawlist", width=width, height=height)


def dashboard():
    """Launch the simulation dashboard GUI.

    Configuration values are loaded from ``config.json``. The ``log_files``
    section is displayed in the dedicated Logging window so it is removed
    from the Parameters window.
    """
    build_graph()
    dpg.create_context()

    import json

    with open(Config.input_path("config.json")) as f:
        config_data = json.load(f)

    # populate the editable graph from disk
    try:
        model = load_graph(Config.input_path("graph.json"))
        set_graph(model)
        set_active_file(Config.input_path("graph.json"))
    except Exception as exc:
        print(f"Failed to load graph for editor: {exc}")

    # remove settings duplicated elsewhere in the GUI
    config_data.pop("log_files", None)
    config_data.pop("tick_rate", None)
    config_data.pop("max_ticks", None)

    with dpg.font_registry():
        font_path = os.path.join(
            os.path.dirname(__file__), "..", "assets", "fonts", "consola.ttf"
        )
        default_font = dpg.add_font(font_path, 20)

    dpg.bind_font(default_font)
    dpg.create_viewport(
        title="CWT Simulation Dashboard", width=800, height=800, resizable=True
    )

    with dpg.window(label="Control Panel", width=800, height=200):
        dpg.add_slider_float(
            label="Tick Rate (sec)",
            default_value=Config.tick_rate,
            min_value=0.1,
            max_value=2.0,
            callback=tick_rate_changed,
            tag="tick_rate_slider",
        )
        dpg.add_input_int(
            label="Max Ticks",
            default_value=Config.max_ticks,
            callback=max_ticks_changed,
            tag="max_ticks_input",
        )
        dpg.add_input_int(
            label="Max Children/Node",
            default_value=Config.max_children_per_node,
            callback=max_children_changed,
            tag="max_children_input",
        )
        dpg.add_button(
            label="Pause", callback=pause_resume_callback, tag="pause_button"
        )
        dpg.add_button(
            label="Start Simulation", callback=start_sim_callback, tag="start_button"
        )
        dpg.add_text("Tick: 0", tag="tick_counter")

    with dpg.window(label="Logging", width=250, height=300):
        for name in sorted(Config.log_files):
            dpg.add_checkbox(
                label=name,
                default_value=Config.log_files[name],
                callback=toggle_log,
                user_data=name,
            )

    with dpg.window(label="Parameters", width=250, height=400):
        _add_param_controls(config_data)

    with dpg.window(label="Graph Editor", width=200, height=150, pos=(610, 120)):
        add_toolbar()
        dpg.add_button(label="Add Node", callback=add_node_callback)
        dpg.add_button(
            label="Add Connection",
            callback=connection_tool.start_add_connection,
        )

    with dpg.window(label="Causal Graph", width=800, height=460, tag="graph_window"):
        with dpg.child_window(tag="graph_child", horizontal_scrollbar=True):
            dpg.add_drawlist(width=1, height=1, tag="graph_drawlist")
            for node_id, (x, y) in node_positions.items():
                color = node_colors[node_id]
                node_tag = dpg.draw_circle(
                    center=(x, y),
                    radius=30,
                    color=color,
                    fill=color,
                    parent="graph_drawlist",
                )
                label_tag = dpg.draw_text(
                    (x - 25, y - 10),
                    f"{node_id}\\nTicks: 0",
                    size=15,
                    tag=f"{node_id}_label",
                    parent="graph_drawlist",
                )
                node_tags[node_id] = node_tag
                tick_counters[node_id] = label_tag

            # Draw directed edges
            dpg.draw_arrow(
                p1=node_positions["A1"],
                p2=node_positions["C"],
                color=(150, 150, 150),
                thickness=2,
                parent="graph_drawlist",
            )
            dpg.draw_arrow(
                p1=node_positions["A2"],
                p2=node_positions["C"],
                color=(150, 150, 150),
                thickness=2,
                parent="graph_drawlist",
            )

    with dpg.window(label="Legend", width=180, height=120, pos=(610, 260)):
        dpg.add_text(
            "\u2b1b SIP_BUD — Stable replication", color=ORIGIN_COLORS["SIP_BUD"]
        )
        dpg.add_text(
            "\u2b1b SIP_RECOMB — Dual-parent recomb", color=ORIGIN_COLORS["SIP_RECOMB"]
        )
        dpg.add_text("\u2b1b CSP — Collapse-seeded", color=ORIGIN_COLORS["CSP"])

    with dpg.item_handler_registry(tag="graph_handlers"):
        dpg.add_item_resize_handler(callback=graph_resize_callback)
    dpg.bind_item_handler_registry("graph_window", "graph_handlers")

    global canvas
    canvas = GraphCanvas(drawlist_tag="graph_drawlist", window_tag="graph_window")

    with dpg.window(
        label="Connection Properties",
        modal=True,
        show=False,
        tag="connection_properties",
        no_close=True,
    ):
        dpg.add_input_text(label="From", tag="from_field")
        dpg.add_input_text(label="To", tag="to_field")
        dpg.add_input_float(label="Delay", default_value=1.0, tag="delay_input")
        dpg.add_input_float(label="Attenuation", default_value=1.0, tag="atten_input")
        dpg.add_combo(
            ["Directed Edge", "Bridge"],
            default_value="Directed Edge",
            tag="conn_type",
        )
        dpg.add_button(label="Save", callback=connection_tool.save_connection)
        dpg.add_button(
            label="Delete",
            callback=connection_tool.delete_connection,
            tag="delete_conn_button",
            show=False,
        )

    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window("graph_window", True)
    graph_resize_callback(None, None, None)
    if canvas is not None:
        canvas.build_full_canvas()
    dpg.start_dearpygui()
    dpg.destroy_context()


def launch() -> None:
    """Entry point for the GUI."""
    dashboard()
