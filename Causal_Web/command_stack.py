"""Undo/redo command stack utilities without GUI dependencies."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

from .graph.model import GraphModel


class Command:
    """Base class for actions that can be undone and redone."""

    def execute(self) -> None:  # pragma: no cover - interface
        """Apply the command."""

    def undo(self) -> None:  # pragma: no cover - interface
        """Reverse the command."""


@dataclass
class CommandStack:
    """Maintain undo and redo stacks for commands."""

    undo_stack: List[Command] = field(default_factory=list)
    redo_stack: List[Command] = field(default_factory=list)

    def do(self, command: Command) -> None:
        """Execute ``command`` and push it onto the undo stack."""

        command.execute()
        self.undo_stack.append(command)
        self.redo_stack.clear()

    def undo(self) -> None:
        """Undo the most recent command if any."""

        if not self.undo_stack:
            return
        cmd = self.undo_stack.pop()
        cmd.undo()
        self.redo_stack.append(cmd)

    def redo(self) -> None:
        """Redo the most recently undone command if any."""

        if not self.redo_stack:
            return
        cmd = self.redo_stack.pop()
        cmd.execute()
        self.undo_stack.append(cmd)


@dataclass
class AddNodeCommand(Command):
    """Command that inserts a new node into a :class:`GraphModel`."""

    model: GraphModel
    node_id: str
    kwargs: dict

    def execute(self) -> None:
        self.model.add_node(self.node_id, **self.kwargs)

    def undo(self) -> None:
        self.model.nodes.pop(self.node_id, None)


@dataclass
class DeleteEdgeCommand(Command):
    """Command that removes an edge from a :class:`GraphModel`."""

    model: GraphModel
    index: int
    connection_type: str = "edge"
    _removed: dict | None = None

    def execute(self) -> None:
        target_list = (
            self.model.edges if self.connection_type == "edge" else self.model.bridges
        )
        if self.index < 0 or self.index >= len(target_list):
            return
        self._removed = target_list[self.index]
        self.model.remove_connection(self.index, self.connection_type)

    def undo(self) -> None:
        if self._removed is None:
            return
        target_list = (
            self.model.edges if self.connection_type == "edge" else self.model.bridges
        )
        target_list.insert(self.index, self._removed)


@dataclass
class MoveNodeCommand(Command):
    """Command that changes a node's ``(x, y)`` position."""

    model: GraphModel
    node_id: str
    new_pos: Tuple[float, float]
    _old_pos: Tuple[float, float] | None = None

    def execute(self) -> None:
        node = self.model.nodes.get(self.node_id)
        if node is None:
            return
        self._old_pos = (node.get("x", 0.0), node.get("y", 0.0))
        node["x"], node["y"] = self.new_pos

    def undo(self) -> None:
        if self._old_pos is None:
            return
        node = self.model.nodes.get(self.node_id)
        if node is None:
            return
        node["x"], node["y"] = self._old_pos


@dataclass
class AddObserverCommand(Command):
    """Command that inserts an observer into a :class:`GraphModel`."""

    model: GraphModel
    observer: dict
    index: int | None = None

    def execute(self) -> None:
        if self.index is None:
            self.model.observers.append(self.observer)
            self.index = len(self.model.observers) - 1
        else:
            self.model.observers.insert(self.index, self.observer)

    def undo(self) -> None:
        if self.index is None:
            return
        if 0 <= self.index < len(self.model.observers):
            self.model.observers.pop(self.index)


@dataclass
class AddMetaNodeCommand(Command):
    """Command that inserts a meta node into a :class:`GraphModel`."""

    model: GraphModel
    meta_id: str
    data: dict

    def execute(self) -> None:
        self.model.meta_nodes[self.meta_id] = dict(self.data)

    def undo(self) -> None:
        self.model.meta_nodes.pop(self.meta_id, None)


@dataclass
class MoveMetaNodeCommand(Command):
    """Command that updates a meta node's position."""

    model: GraphModel
    meta_id: str
    new_pos: Tuple[float, float]
    _old_pos: Tuple[float, float] | None = None

    def execute(self) -> None:
        data = self.model.meta_nodes.get(self.meta_id)
        if data is None:
            return
        self._old_pos = (data.get("x", 0.0), data.get("y", 0.0))
        data["x"], data["y"] = self.new_pos

    def undo(self) -> None:
        if self._old_pos is None:
            return
        data = self.model.meta_nodes.get(self.meta_id)
        if data is None:
            return
        data["x"], data["y"] = self._old_pos
