"""Store for static graph and latest snapshot delta."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from PySide6.QtCore import QObject, Property, Signal, Slot

from ..ipc import Client


@dataclass(frozen=True)
class GraphDTO:
    """Deterministic representation of a graph.

    The ordering of nodes and edges is stable so the resulting ``dict`` is
    suitable for hashing or comparison.
    """

    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    observers: List[Dict[str, Any]] = field(default_factory=list)
    bridges: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Return the DTO as a plain ``dict``."""
        return {
            "nodes": self.nodes,
            "edges": self.edges,
            "observers": self.observers,
            "bridges": self.bridges,
        }


class Store(QObject):
    """Maintain static graph data and the latest delta."""

    selectionChanged = Signal(int)
    edgeSelectionChanged = Signal(int)
    observerSelectionChanged = Signal(int)
    bridgeSelectionChanged = Signal(int)

    def __init__(self) -> None:
        super().__init__()
        self.graph_static: Dict[str, Any] = {
            "nodes": [],
            "edges": [],
            "observers": [],
            "bridges": [],
        }
        self.latest_delta: Dict[str, Any] = {}
        self._selected: int = -1
        self._selected_edge: int = -1
        self._selected_observer: int = -1
        self._selected_bridge: int = -1
        self._next_id = 0
        self._client: Optional[Client] = None

    @Property(int, notify=selectionChanged)
    def selectedNode(self) -> int:
        """Return the currently selected node id or ``-1`` when none."""

        return self._selected

    @selectedNode.setter
    def selectedNode(self, value: int) -> None:
        value = int(value)
        if self._selected != value:
            self._selected = value
            if value >= 0:
                self._selected_edge = -1
                self._selected_observer = -1
                self._selected_bridge = -1
            self.selectionChanged.emit(value)

    @Property(int, notify=edgeSelectionChanged)
    def selectedEdge(self) -> int:
        """Return the currently selected edge index or ``-1`` when none."""

        return self._selected_edge

    @selectedEdge.setter
    def selectedEdge(self, value: int) -> None:
        value = int(value)
        if self._selected_edge != value:
            self._selected_edge = value
            if value >= 0:
                self._selected = -1
                self._selected_observer = -1
                self._selected_bridge = -1
            self.edgeSelectionChanged.emit(value)

    @Property(int, notify=observerSelectionChanged)
    def selectedObserver(self) -> int:
        """Return the currently selected observer index or ``-1`` when none."""

        return self._selected_observer

    @selectedObserver.setter
    def selectedObserver(self, value: int) -> None:
        value = int(value)
        if self._selected_observer != value:
            self._selected_observer = value
            if value >= 0:
                self._selected = -1
                self._selected_edge = -1
                self._selected_bridge = -1
            self.observerSelectionChanged.emit(value)

    @Property(int, notify=bridgeSelectionChanged)
    def selectedBridge(self) -> int:
        """Return the currently selected bridge index or ``-1`` when none."""

        return self._selected_bridge

    @selectedBridge.setter
    def selectedBridge(self, value: int) -> None:
        value = int(value)
        if self._selected_bridge != value:
            self._selected_bridge = value
            if value >= 0:
                self._selected = -1
                self._selected_edge = -1
                self._selected_observer = -1
            self.bridgeSelectionChanged.emit(value)

    def set_static(self, data: Dict[str, Any]) -> None:
        """Set immutable static graph information."""
        self.graph_static = data
        self.latest_delta.clear()

    def apply_delta(self, delta: Dict[str, Any]) -> None:
        """Coalesce ``delta`` into ``latest_delta`` so only the newest values remain."""
        for key, value in delta.items():
            if (
                key in self.latest_delta
                and isinstance(self.latest_delta[key], dict)
                and isinstance(value, dict)
            ):
                self.latest_delta[key].update(value)
            else:
                self.latest_delta[key] = value

    def current_state(self) -> Dict[str, Any]:
        """Return a merged view of static graph and latest delta."""
        state = self.graph_static.copy()
        state.update(self.latest_delta)
        return state

    @Slot()
    def load_graph(self) -> None:
        """Compile and send the current graph to the engine."""
        if self._client:
            dto = self.compile_graph()
            asyncio.create_task(self._client.send({"cmd": "load_graph", "graph": dto}))

    def set_client(self, client: Client) -> None:
        """Attach a WebSocket ``client`` for control messages."""
        self._client = client

    @Slot(result=dict)
    def graph_arrays(self) -> Dict[str, Any]:
        """Return arrays for :class:`GraphView` including observers and bridges."""

        nodes: List[Tuple[float, float]] = []
        labels: List[str] = []
        colors: List[str] = []
        flags: List[bool] = []

        for n in self.graph_static.get("nodes", []):
            nodes.append((n.get("x", 0.0), n.get("y", 0.0)))
            labels.append(n.get("label", ""))
            colors.append("white")
            flags.append(True)

        for o in self.graph_static.get("observers", []):
            nodes.append((o.get("x", 0.0), o.get("y", 0.0)))
            labels.append(o.get("id", ""))
            colors.append("orange")
            flags.append(True)

        edges = [
            (e.get("from"), e.get("to")) for e in self.graph_static.get("edges", [])
        ]
        edges.extend(
            (b.get("from"), b.get("to")) for b in self.graph_static.get("bridges", [])
        )
        return {
            "nodes": nodes,
            "edges": edges,
            "labels": labels,
            "colors": colors,
            "flags": flags,
        }

    @Slot(result=dict)
    def compile_graph(self) -> Dict[str, Any]:
        """Compile current state into a deterministic graph ``dict``.

        Nodes are sorted by their ``id`` field and edges are sorted by their
        ``from``/``to`` identifiers so that repeated compilations of the same
        graph yield identical DTOs.
        """

        state = self.current_state()
        nodes = sorted(state.get("nodes", []), key=lambda n: n.get("id"))
        edges = sorted(
            state.get("edges", []), key=lambda e: (e.get("from"), e.get("to"))
        )
        observers = sorted(state.get("observers", []), key=lambda o: o.get("id", ""))
        bridges = sorted(
            state.get("bridges", []), key=lambda b: (b.get("from"), b.get("to"))
        )
        dto = GraphDTO(nodes=nodes, edges=edges, observers=observers, bridges=bridges)
        return dto.to_dict()

    @Slot(float, float, result=int)
    def add_node(self, x: float, y: float) -> int:
        """Add a node at ``(x, y)`` returning its identifier."""

        node_id = self._next_id
        self._next_id += 1
        self.graph_static.setdefault("nodes", []).append(
            {"id": node_id, "x": x, "y": y}
        )
        return node_id

    @Slot(int)
    def delete_node(self, node_id: int) -> None:
        """Remove ``node_id`` and any incident edges."""

        nodes = self.graph_static.get("nodes", [])
        self.graph_static["nodes"] = [n for n in nodes if n.get("id") != node_id]
        edges = self.graph_static.get("edges", [])
        self.graph_static["edges"] = [
            e for e in edges if e.get("from") != node_id and e.get("to") != node_id
        ]
        if self._selected == node_id:
            self.selectedNode = -1

    @Slot(int, int)
    def connect_nodes(self, a: int, b: int) -> None:
        """Create an edge from ``a`` to ``b``."""
        self.graph_static.setdefault("edges", []).append(
            {"from": int(a), "to": int(b), "delay": 1}
        )

    @Slot(int, float, float)
    def move_node(self, node_id: int, x: float, y: float) -> None:
        """Update position of ``node_id``."""

        for node in self.graph_static.get("nodes", []):
            if node.get("id") == node_id:
                node["x"] = x
                node["y"] = y
                break

    @Slot(int, str)
    def set_node_label(self, node_id: int, label: str) -> None:
        """Set ``label`` on ``node_id`` if it exists."""

        for node in self.graph_static.get("nodes", []):
            if node.get("id") == node_id:
                node["label"] = label
                break

    @Slot(int, result=dict)
    def get_node(self, node_id: int) -> Dict[str, Any]:
        """Return node dictionary for ``node_id`` or an empty dict."""

        for node in self.graph_static.get("nodes", []):
            if node.get("id") == node_id:
                return node
        return {}

    @Slot(int, result=dict)
    def get_edge(self, index: int) -> Dict[str, Any]:
        """Return edge dictionary at ``index`` or an empty dict."""

        edges = self.graph_static.get("edges", [])
        if 0 <= index < len(edges):
            return edges[index]
        return {}

    @Slot(int, int)
    def set_edge_delay(self, index: int, delay: int) -> None:
        """Set ``delay`` on edge at ``index`` if valid."""

        edges = self.graph_static.get("edges", [])
        if 0 <= index < len(edges):
            edges[index]["delay"] = int(delay)

    @Slot(int, result=dict)
    def get_observer(self, index: int) -> Dict[str, Any]:
        """Return observer dictionary at ``index`` or an empty dict."""

        observers = self.graph_static.get("observers", [])
        if 0 <= index < len(observers):
            return observers[index]
        return {}

    @Slot(int, str)
    def set_observer_id(self, index: int, value: str) -> None:
        """Set ``id`` on observer at ``index`` if valid."""

        observers = self.graph_static.get("observers", [])
        if 0 <= index < len(observers):
            observers[index]["id"] = value

    @Slot(int, float)
    def set_observer_frequency(self, index: int, freq: float) -> None:
        """Set ``frequency`` on observer at ``index`` if valid."""

        observers = self.graph_static.get("observers", [])
        if 0 <= index < len(observers):
            observers[index]["frequency"] = float(freq)

    @Slot(int, result=dict)
    def get_bridge(self, index: int) -> Dict[str, Any]:
        """Return bridge dictionary at ``index`` or an empty dict."""

        bridges = self.graph_static.get("bridges", [])
        if 0 <= index < len(bridges):
            return bridges[index]
        return {}

    @Slot(int, bool)
    def set_bridge_entangled(self, index: int, value: bool) -> None:
        """Set ``is_entangled`` on bridge at ``index`` if valid."""

        bridges = self.graph_static.get("bridges", [])
        if 0 <= index < len(bridges):
            bridges[index]["is_entangled"] = bool(value)

    @Slot(float, float, result=int)
    def find_node(self, x: float, y: float) -> int:
        """Return id of node nearest to ``(x, y)`` or ``-1`` if none."""

        best = (-1, float("inf"))
        for node in self.graph_static.get("nodes", []):
            dx = node.get("x", 0.0) - x
            dy = node.get("y", 0.0) - y
            d = dx * dx + dy * dy
            if d < best[1]:
                best = (node.get("id", -1), d)
        return best[0]

    @Slot(float, float, result=int)
    def find_edge(self, x: float, y: float) -> int:
        """Return index of edge nearest to ``(x, y)`` or ``-1`` if none."""

        nodes = {n.get("id"): n for n in self.graph_static.get("nodes", [])}
        best = (-1, float("inf"))
        for idx, edge in enumerate(self.graph_static.get("edges", [])):
            a = nodes.get(edge.get("from"))
            b = nodes.get(edge.get("to"))
            if not a or not b:
                continue
            dist = self._edge_distance(x, y, a, b)
            if dist < best[1]:
                best = (idx, dist)
        return best[0]

    @Slot(float, float, result=int)
    def find_observer(self, x: float, y: float) -> int:
        """Return index of observer nearest to ``(x, y)`` or ``-1`` if none."""

        best = (-1, float("inf"))
        for idx, obs in enumerate(self.graph_static.get("observers", [])):
            dx = obs.get("x", 0.0) - x
            dy = obs.get("y", 0.0) - y
            d = dx * dx + dy * dy
            if d < best[1]:
                best = (idx, d)
        return best[0]

    @Slot(float, float, result=int)
    def find_bridge(self, x: float, y: float) -> int:
        """Return index of bridge nearest to ``(x, y)`` or ``-1`` if none."""

        nodes = {n.get("id"): n for n in self.graph_static.get("nodes", [])}
        best = (-1, float("inf"))
        for idx, bridge in enumerate(self.graph_static.get("bridges", [])):
            a = nodes.get(bridge.get("from"))
            b = nodes.get(bridge.get("to"))
            if not a or not b:
                continue
            dist = self._edge_distance(x, y, a, b)
            if dist < best[1]:
                best = (idx, dist)
        return best[0]

    def _edge_distance(
        self, x: float, y: float, a: Dict[str, Any], b: Dict[str, Any]
    ) -> float:
        """Return squared distance from point ``(x, y)`` to segment AB."""

        ax, ay = a.get("x", 0.0), a.get("y", 0.0)
        bx, by = b.get("x", 0.0), b.get("y", 0.0)
        px, py = x - ax, y - ay
        dx, dy = bx - ax, by - ay
        denom = dx * dx + dy * dy
        if denom > 0.0:
            t = max(0.0, min(1.0, (px * dx + py * dy) / denom))
            nx = ax + t * dx
            ny = ay + t * dy
        else:
            nx, ny = ax, ay
        dx2 = x - nx
        dy2 = y - ny
        return dx2 * dx2 + dy2 * dy2

    @Slot(result=list)
    def validate_graph(self) -> List[str]:
        """Return validation warnings for duplicates, self-loops or missing props."""

        warnings: List[str] = []
        ids: Set[int] = set()
        for node in self.graph_static.get("nodes", []):
            nid = node.get("id")
            if nid in ids:
                warnings.append(f"duplicate node id {nid}")
            else:
                ids.add(nid)
            if any(k not in node for k in ("id", "x", "y")):
                warnings.append(f"node {nid} missing properties")

        edge_pairs: Set[Tuple[int, int]] = set()
        for edge in self.graph_static.get("edges", []):
            a = edge.get("from")
            b = edge.get("to")
            if a is None or b is None:
                warnings.append("edge missing endpoint")
                continue
            if a == b:
                warnings.append(f"self-loop on {a}")
            pair = (int(a), int(b))
            if pair in edge_pairs:
                warnings.append(f"duplicate edge {a}->{b}")
            else:
                edge_pairs.add(pair)
        return warnings
