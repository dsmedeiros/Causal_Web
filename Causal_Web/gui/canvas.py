# canvas.py

"""Interactive canvas for displaying :class:`GraphModel` structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional

import dearpygui.dearpygui as dpg

from .state import (
    get_graph,
    set_selected_node,
    get_selected_node,
)
from . import connection_tool
from .command_stack import CommandStack

_commands = CommandStack()


@dataclass
class GraphCanvas:
    """Widget for drawing graphs using Dear PyGui.

    The canvas creates its own item handler registry to manage mouse events
    which allows nodes to be dragged and selected without errors on startup.
    """

    drawlist_tag: str = "graph_canvas_drawlist"
    window_tag: str = "graph_canvas_window"
    dragging_node: Optional[str] = field(default=None, init=False)

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
        # event handlers must be attached via an item handler registry so that
        # they remain valid across Dear PyGui versions
        with dpg.item_handler_registry(tag=f"{self.drawlist_tag}_handlers") as h:
            dpg.add_mouse_down_handler(
                button=dpg.mvMouseButton_Left,
                callback=self._handle_mouse_down,
                parent=h,
            )
            dpg.add_mouse_release_handler(
                button=dpg.mvMouseButton_Left,
                callback=self._handle_click,
                parent=h,
            )
        dpg.bind_item_handler_registry(self.drawlist_tag, h)

    def redraw(self) -> None:
        """Clear and redraw the entire graph."""
        if not dpg.does_item_exist(self.drawlist_tag):
            return
        self._update_drag()
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

    def _handle_mouse_down(self, sender, app_data):
        """Begin dragging the node under the cursor if any."""
        mouse = dpg.get_mouse_pos(local=False)
        origin = dpg.get_item_rect_min(self.drawlist_tag)
        for node_id, item in self.node_items.items():
            center = dpg.get_item_configuration(item)["center"]
            dx = mouse[0] - (origin[0] + center[0])
            dy = mouse[1] - (origin[1] + center[1])
            if dx * dx + dy * dy <= 20 * 20:
                print(f"[GraphCanvas] Selected node {node_id}")
                set_selected_node(node_id)
                self.dragging_node = node_id
                return
        set_selected_node(None)

    def _handle_click(self, sender, app_data):
        """Handle mouse release events over nodes."""
        mouse = dpg.get_mouse_pos(local=False)
        origin = dpg.get_item_rect_min(self.drawlist_tag)
        print(f"[GraphCanvas] Click at {mouse}")
        for node_id, item in self.node_items.items():
            center = dpg.get_item_configuration(item)["center"]
            dx = mouse[0] - (origin[0] + center[0])
            dy = mouse[1] - (origin[1] + center[1])
            if dx * dx + dy * dy <= 20 * 20:
                if connection_tool.handle_node_click(node_id):
                    print(f"[GraphCanvas] Connection tool handled click on {node_id}")
                return
        print("[GraphCanvas] Click not on any node")
        set_selected_node(None)

    def _update_drag(self) -> None:
        """Move the selected node when the mouse is dragged."""
        if self.dragging_node is None:
            return
        if dpg.is_mouse_button_down(dpg.mvMouseButton_Left):
            graph = get_graph()
            mouse = dpg.get_mouse_pos(local=False)
            origin = dpg.get_item_rect_min(self.drawlist_tag)
            x = mouse[0] - origin[0]
            y = mouse[1] - origin[1]
            node = graph.nodes.get(self.dragging_node)
            if node is not None:
                node["x"] = x
                node["y"] = y
                print(f"[GraphCanvas] Dragging {self.dragging_node} to {(x, y)}")
        else:
            print(f"[GraphCanvas] Finished dragging {self.dragging_node}")
            self.dragging_node = None

    def auto_layout(self) -> None:
        """Apply a force-directed layout to the current graph."""

        graph = get_graph()
        graph.apply_spring_layout()
        self.redraw()

    def undo(self) -> None:
        _commands.undo()
        self.redraw()

    def redo(self) -> None:
        _commands.redo()
        self.redraw()
