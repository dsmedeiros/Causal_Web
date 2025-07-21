import os
import dearpygui.dearpygui as dpg

from .canvas import GraphCanvas
from ..graph.io import load_graph, save_graph
from .state import get_graph, set_graph, set_active_file
from ..config import Config

_last_directory = Config.input_dir


def _get_canvas() -> GraphCanvas | None:
    if dpg.does_item_exist("graph_canvas_drawlist"):
        return dpg.get_item_user_data("graph_canvas_drawlist")
    return None


def _load_dialog_callback(sender, app_data):
    global _last_directory
    path = app_data["file_path_name"]
    if not path:
        return
    _last_directory = os.path.dirname(path)
    try:
        graph = load_graph(path)
    except Exception as exc:
        print(f"Failed to load graph: {exc}")
        return
    set_graph(graph)
    set_active_file(path)


def _save_dialog_callback(sender, app_data):
    global _last_directory
    path = app_data["file_path_name"]
    if not path:
        return
    _last_directory = os.path.dirname(path)
    try:
        save_graph(path, get_graph())
    except Exception as exc:
        print(f"Failed to save graph: {exc}")
        return
    set_active_file(path)


def add_toolbar() -> None:
    """Create toolbar buttons for load/save/new graph actions."""
    with dpg.group(horizontal=True, tag="graph_toolbar"):
        dpg.add_button(
            label="Load", callback=lambda: dpg.show_item("graph_load_dialog")
        )
        dpg.add_button(
            label="Save", callback=lambda: dpg.show_item("graph_save_dialog")
        )
        dpg.add_button(label="New", callback=lambda: (_new_graph(),))
        dpg.add_button(label="Auto Layout", callback=_auto_layout)
        dpg.add_button(label="Undo", callback=_undo)
        dpg.add_button(label="Redo", callback=_redo)

    dpg.add_file_dialog(
        directory_selector=False,
        show=False,
        callback=_load_dialog_callback,
        tag="graph_load_dialog",
        default_path=_last_directory,
    )
    dpg.add_file_dialog(
        directory_selector=False,
        show=False,
        callback=_save_dialog_callback,
        tag="graph_save_dialog",
        default_filename="graph.json",
        default_path=_last_directory,
    )


def _new_graph():
    from ..graph.io import new_graph

    set_graph(new_graph(True))
    set_active_file(None)


def _auto_layout():
    canvas = _get_canvas()
    if canvas is not None:
        canvas.auto_layout()


def _undo():
    canvas = _get_canvas()
    if canvas is not None:
        canvas.undo()


def _redo():
    canvas = _get_canvas()
    if canvas is not None:
        canvas.redo()
