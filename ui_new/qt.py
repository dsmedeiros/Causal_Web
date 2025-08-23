"""Qt compatibility layer for headless test environments.

This module attempts to import the real PySide6 bindings. If the import
fails (for example, when the system ``libGL`` library is unavailable), it
falls back to lightweight stub implementations that provide just enough
behaviour for the unit tests. The stubs implement the subset of the QtCore
API used throughout the ``ui_new.state`` package.
"""

from __future__ import annotations

try:  # pragma: no cover - exercised indirectly by tests
    from PySide6.QtCore import (
        QFileSystemWatcher,
        QObject,
        Property,
        QElapsedTimer,
        QStringListModel,
        Signal,
        Slot,
    )
except Exception:  # pragma: no cover - used when Qt cannot be imported

    class QObject:  # type: ignore[no-redef]
        """Minimal stand‑in for :class:`PySide6.QtCore.QObject`."""

        def __init__(self, *_, **__):
            pass

    class Signal:  # type: ignore[no-redef]
        """Simple replacement that ignores all connections."""

        def __init__(self, *_, **__):
            pass

        def emit(self, *_, **__):
            pass

    def Slot(*_args, **_kwargs):  # type: ignore[no-redef]
        """Decorator that returns the function unchanged."""

        def decorator(func):
            return func

        return decorator

    def Property(*_args, **_kwargs):  # type: ignore[no-redef]
        """Return a basic :class:`property` descriptor."""

        def wrapper(fget=None, fset=None, freset=None):
            return property(fget, fset, freset)

        return wrapper

    class QStringListModel(list):  # type: ignore[no-redef]
        """List-backed stub used for ``LogsModel`` tests."""

        def setStringList(self, items):
            self[:] = list(items)

    class QElapsedTimer:  # type: ignore[no-redef]
        """Timer stub returning zero elapsed time."""

        def restart(self) -> None:
            pass

        def elapsed(self) -> int:
            return 0

    class QFileSystemWatcher:  # type: ignore[no-redef]
        """Watcher stub that satisfies the ``GAModel`` constructor."""

        def __init__(self, *_, **__):
            pass


__all__ = [
    "QObject",
    "Property",
    "Signal",
    "Slot",
    "QStringListModel",
    "QElapsedTimer",
    "QFileSystemWatcher",
]
