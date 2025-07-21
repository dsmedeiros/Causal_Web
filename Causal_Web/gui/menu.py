"""Menu helpers for loading, saving and creating graphs."""

from __future__ import annotations

import os
import dearpygui.dearpygui as dpg

from ..config import Config
from ..graph.io import load_graph, new_graph, save_graph
from .state import get_graph, set_graph


_last_directory = Config.input_dir


def _open_load_dialog():
    dpg.show_item("graph_load_dialog")


def _open_save_dialog():
    dpg.show_item("graph_save_dialog")


def _on_load_dialog(sender, app_data):
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


def _on_save_dialog(sender, app_data):
    global _last_directory
    path = app_data["file_path_name"]
    if not path:
        return
    _last_directory = os.path.dirname(path)
    try:
        save_graph(path, get_graph())
    except Exception as exc:
        print(f"Failed to save graph: {exc}")


def _new_graph_callback():
    set_graph(new_graph(True))


def add_file_menu() -> None:
    """Create the file menu with Load/Save/New actions."""
    with dpg.menu(label="File"):
        dpg.add_menu_item(label="Load Graph", callback=_open_load_dialog)
        dpg.add_menu_item(label="Save Graph", callback=_open_save_dialog)
        dpg.add_menu_item(label="New Graph", callback=_new_graph_callback)
    dpg.add_file_dialog(
        directory_selector=False,
        show=False,
        callback=_on_load_dialog,
        tag="graph_load_dialog",
        default_path=_last_directory,
    )
    dpg.add_file_dialog(
        directory_selector=False,
        show=False,
        callback=_on_save_dialog,
        tag="graph_save_dialog",
        default_filename="graph.json",
        default_path=_last_directory,
    )
