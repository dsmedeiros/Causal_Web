"""Shared helper mixins for GUI dock panels."""

from __future__ import annotations

from typing import Any


class PanelMixin:
    """Provide common behaviour for dock panels."""

    dirty: bool

    def _minimize(self) -> None:
        """Hide the panel when focus is lost."""
        self.hide()

    def _mark_dirty(self, *args: Any) -> None:
        """Flag that the panel has unsaved changes."""
        self.dirty = True
