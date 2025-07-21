# canvas.py

"""Interactive canvas for displaying :class:`GraphModel` structures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import dearpygui.dearpygui as dpg

from .state import (
    get_graph,
    set_selected_node,
    get_selected_node,
)
from . import connection_tool


@dataclass
class GraphCanvas:
    """Widget for drawing graphs using Dear PyGui."""

    drawlist_tag: str = "graph_canvas_drawlist"
    window_tag: str = "graph_canvas_window"

    def __post_init__(self) -> None:
        """Create the Dear PyGui widgets backing the canvas."""
        if not dpg.does_item_exist(self.window_tag):
            with dpg.window(label="Graph View", tag=self.window_tag):
                dpg.add_drawlist(tag=self.drawlist_tag, width=600, height=400)
                dpg.add_text("", tag="graph_status_bar")
        self.node_items: Dict[str, int] = {}
        self.edge_items: list[int] = []
        self.bridge_items: list[int] = []
        dpg.set_item_user_data(self.drawlist_tag, self)
        dpg.set_item_callback(self.drawlist_tag, self._handle_click)

    def redraw(self) -> None:
        """Clear and redraw the entire graph."""
        if not dpg.does_item_exist(self.drawlist_tag):
            return
        dpg.delete_item(self.drawlist_tag, children_only=True)
        graph = get_graph()
        self.node_items.clear()
        self.edge_items.clear()
        self.bridge_items.clear()

        for edge in graph.edges:
            p1 = graph.node_position(edge.get("from"))
            p2 = graph.node_position(edge.get("to"))
            if p1 is None or p2 is None:
                continue
            item = dpg.draw_arrow(
                p1=p1, p2=p2, color=(150, 150, 150), parent=self.drawlist_tag
            )
            self.edge_items.append(item)

        for bridge in graph.bridges:
            nodes = bridge.get("nodes")
            if not nodes or len(nodes) != 2:
                continue
            p1 = graph.node_position(nodes[0])
            p2 = graph.node_position(nodes[1])
            if p1 is None or p2 is None:
                continue
            item = dpg.draw_line(
                p1=p1,
                p2=p2,
                color=(200, 100, 100),
                thickness=1,
                parent=self.drawlist_tag,
            )
            self.bridge_items.append(item)

        for node_id, data in graph.nodes.items():
            pos = graph.node_position(node_id)
            if pos is None:
                continue
            x, y = pos
            item = dpg.draw_circle(
                center=(x, y),
                radius=20,
                color=(200, 200, 200),
                fill=(60, 60, 60),
                parent=self.drawlist_tag,
                tag=f"node_{node_id}",
            )
            dpg.draw_text(pos=(x - 10, y - 5), text=node_id, parent=self.drawlist_tag)
            self.node_items[node_id] = item

    def _handle_click(self, sender, app_data):
        """Select a node if the click occurred over one."""
        mouse = dpg.get_mouse_pos(local=False)
        for node_id, item in self.node_items.items():
            center = dpg.get_item_configuration(item)["center"]
            dx = mouse[0] - center[0]
            dy = mouse[1] - center[1]
            if dx * dx + dy * dy <= 20 * 20:
                if connection_tool.handle_node_click(node_id):
                    return
                set_selected_node(node_id)
                return
        set_selected_node(None)
