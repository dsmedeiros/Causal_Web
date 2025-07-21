"""Undo/redo command stack utilities for the GUI."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


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

    model: "GraphModel"
    node_id: str
    kwargs: dict

    def execute(self) -> None:
        self.model.add_node(self.node_id, **self.kwargs)

    def undo(self) -> None:
        self.model.nodes.pop(self.node_id, None)
