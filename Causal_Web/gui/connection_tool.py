"""Utilities for adding and editing graph connections within the GUI."""

from __future__ import annotations

from typing import Optional

import dearpygui.dearpygui as dpg

from .state import get_graph, set_selected_connection


_add_mode = False
_source: Optional[str] = None
_target: Optional[str] = None
_edit: Optional[tuple[str, int]] = None


def handle_node_click(node_id: str) -> bool:
    """Handle node click events during add-connection mode.

    Returns ``True`` if the click was consumed by the tool.
    """
    global _add_mode, _source, _target
    if not _add_mode:
        return False
    if _source is None:
        _source = node_id
        return True
    _target = node_id
    _show_properties(edit=False)
    _add_mode = False
    return True


def start_add_connection() -> None:
    """Enable interactive connection creation."""
    global _add_mode, _source, _target, _edit
    _add_mode = True
    _source = None
    _target = None
    _edit = None
    dpg.configure_item("connection_properties", show=False)


def edit_connection(conn_type: str, index: int) -> None:
    """Open the properties panel for an existing connection."""
    global _edit
    _edit = (conn_type, index)
    _populate_fields(conn_type, index)
    _show_properties(edit=True)


def _populate_fields(conn_type: str, index: int) -> None:
    graph = get_graph()
    data = graph.edges[index] if conn_type == "edge" else graph.bridges[index]
    if conn_type == "edge":
        dpg.set_value("conn_type", "Directed Edge")
        dpg.set_value("delay_input", float(data.get("delay", 1)))
        dpg.set_value("atten_input", float(data.get("attenuation", 1)))
        dpg.set_value("from_field", data.get("from"))
        dpg.set_value("to_field", data.get("to"))
    else:
        dpg.set_value("conn_type", "Bridge")
        dpg.set_value("delay_input", float(data.get("delay", 1)))
        dpg.set_value("atten_input", float(data.get("attenuation", 1)))
        nodes = data.get("nodes", ["", ""])
        dpg.set_value("from_field", nodes[0])
        dpg.set_value("to_field", nodes[1])


def _show_properties(edit: bool) -> None:
    if _source is not None:
        dpg.set_value("from_field", _source)
    if _target is not None:
        dpg.set_value("to_field", _target)
    dpg.configure_item("delete_conn_button", show=edit)
    dpg.show_item("connection_properties")


def save_connection(sender, app_data, user_data):
    graph = get_graph()
    conn_type = dpg.get_value("conn_type")
    delay = float(dpg.get_value("delay_input"))
    atten = float(dpg.get_value("atten_input"))
    source = dpg.get_value("from_field")
    target = dpg.get_value("to_field")
    type_key = "edge" if conn_type == "Directed Edge" else "bridge"

    if _edit is not None:
        etype, idx = _edit
        graph.update_connection(idx, etype, delay=delay, attenuation=atten)
        set_selected_connection((etype, idx))
    else:
        graph.add_connection(
            source, target, delay=delay, attenuation=atten, connection_type=type_key
        )
        idx = len(graph.edges) - 1 if type_key == "edge" else len(graph.bridges) - 1
        set_selected_connection((type_key, idx))
    dpg.configure_item("connection_properties", show=False)


def delete_connection(sender, app_data, user_data):
    if _edit is None:
        return
    graph = get_graph()
    etype, idx = _edit
    graph.remove_connection(idx, etype)
    set_selected_connection(None)
    dpg.configure_item("connection_properties", show=False)
