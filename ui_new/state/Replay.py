from __future__ import annotations

"""Replay model exposed to QML panels."""

import asyncio
from typing import List, Optional, Tuple

from ..qt import QObject, Property, Signal, Slot

from ..ipc import Client

# Global client reference for module-level helpers
client: Optional[Client] = None


class ReplayModel(QObject):
    """Track replay progress as a fraction [0, 1]."""

    progressChanged = Signal(float)
    bookmarksChanged = Signal()
    annotationsChanged = Signal()

    def __init__(self) -> None:
        super().__init__()
        self._progress = 0.0
        self._client: Optional[Client] = None
        self._bookmarks: List[Tuple[str, float]] = []
        self._annotations: List[Tuple[float, str]] = []

    # ------------------------------------------------------------------
    def _get_progress(self) -> float:
        return self._progress

    def _set_progress(self, value: float) -> None:
        if self._progress != value:
            self._progress = value
            self.progressChanged.emit(value)

    progress = Property(float, _get_progress, _set_progress, notify=progressChanged)

    def _get_bookmarks(self) -> List[Dict[str, float]]:
        """Expose bookmarks as a list of mappings for QML."""
        return [{"name": n, "progress": p} for n, p in self._bookmarks]

    bookmarks = Property("QVariant", _get_bookmarks, notify=bookmarksChanged)

    def _get_annotations(self) -> List[Dict[str, Union[float, str]]]:
        """Expose frame annotations for QML."""
        return [{"progress": p, "text": t} for p, t in self._annotations]

    annotations = Property("QVariant", _get_annotations, notify=annotationsChanged)

    # ------------------------------------------------------------------
    def update_progress(self, value: float) -> None:
        """Set the current replay progress."""
        self._set_progress(value)

    # ------------------------------------------------------------------
    def set_client(self, client_in: Client | None) -> None:
        """Attach a WebSocket ``client`` for sending control messages."""
        global client
        self._client = client_in
        client = client_in

    # ------------------------------------------------------------------
    @Slot()
    def play(self) -> None:
        """Start or resume the replay."""
        if self._client:
            asyncio.create_task(
                self._client.send({"ReplayControl": {"action": "play"}})
            )

    @Slot()
    def pause(self) -> None:
        """Pause the replay."""
        if self._client:
            asyncio.create_task(
                self._client.send({"ReplayControl": {"action": "pause"}})
            )

    @Slot(float)
    def seek(self, value: float) -> None:
        """Seek the replay to ``value`` between 0 and 1."""
        self._set_progress(value)
        if self._client:
            asyncio.create_task(
                self._client.send(
                    {"ReplayControl": {"action": "seek", "progress": float(value)}}
                )
            )

    # ------------------------------------------------------------------
    @Slot(str)
    def addBookmark(self, name: str) -> None:
        """Record a bookmark at the current progress."""
        self._bookmarks.append((name, self._progress))
        self.bookmarksChanged.emit()

    @Slot(str)
    def addAnnotation(self, text: str) -> None:
        """Annotate the current frame with ``text``."""
        self._annotations.append((self._progress, text))
        self.annotationsChanged.emit()

    @Slot(str)
    def load(self, dir_path: str) -> None:
        """Load replay data from ``dir_path`` directory."""
        if self._client:
            asyncio.create_task(
                self._client.send(
                    {"ReplayControl": {"action": "load", "path": dir_path}}
                )
            )


# ---------------------------------------------------------------------------
async def open_replay(run_dir: str) -> None:
    """Load a replay from ``run_dir`` and immediately start playback."""
    if client is None:
        raise RuntimeError("client not set")
    await client.send({"type": "ReplayControl", "cmd": "load", "dir": run_dir})
    await client.send({"type": "ReplayControl", "cmd": "play"})
