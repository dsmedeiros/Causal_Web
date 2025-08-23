from __future__ import annotations

"""Logs model exposing entries to QML."""

from ..qt import QStringListModel, Slot


class LogsModel(QStringListModel):
    """Model containing log entries for display."""

    def __init__(self) -> None:
        super().__init__([])

    # ------------------------------------------------------------------
    @Slot(str)
    def add_entry(self, entry: str) -> None:
        """Append ``entry`` to the log list."""
        row = self.rowCount()
        self.insertRow(row)
        self.setData(self.index(row, 0), entry)

    @Slot()
    def clear(self) -> None:
        """Remove all log entries."""
        self.setStringList([])
