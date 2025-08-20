from __future__ import annotations

"""Model for comparing two runs and reporting metric deltas."""

from collections import defaultdict
from pathlib import Path
from typing import Dict, List
import csv

from PySide6.QtCore import QObject, Property, Signal, Slot


class CompareModel(QObject):
    """Load metrics and frames from two run directories for visual comparison."""

    metricDeltaChanged = Signal()
    frameChanged = Signal()
    frameCountChanged = Signal()
    frameIndexChanged = Signal()

    def __init__(self) -> None:
        super().__init__()
        self._metric_delta: Dict[str, float] = {}
        self._frames_a: List[str] = []
        self._frames_b: List[str] = []
        self._index = 0

    # ------------------------------------------------------------------
    def _get_metric_delta(self) -> list[dict[str, float]]:
        """Return metric deltas as a list for QML."""
        return [
            {"category": k, "delta": v} for k, v in sorted(self._metric_delta.items())
        ]

    metricDelta = Property("QVariant", _get_metric_delta, notify=metricDeltaChanged)

    # ------------------------------------------------------------------
    def _get_frame_count(self) -> int:
        """Total number of comparable frames."""
        return min(len(self._frames_a), len(self._frames_b))

    frameCount = Property(int, _get_frame_count, notify=frameCountChanged)

    def _get_frame_index(self) -> int:
        """Current frame index."""
        return self._index

    frameIndex = Property(int, _get_frame_index, notify=frameIndexChanged)

    def _get_frame_a(self) -> str:
        """Path to the current frame from run A."""
        if 0 <= self._index < len(self._frames_a):
            return self._frames_a[self._index]
        return ""

    frameA = Property(str, _get_frame_a, notify=frameChanged)

    def _get_frame_b(self) -> str:
        """Path to the current frame from run B."""
        if 0 <= self._index < len(self._frames_b):
            return self._frames_b[self._index]
        return ""

    frameB = Property(str, _get_frame_b, notify=frameChanged)

    # ------------------------------------------------------------------
    @Slot(int)
    def setFrame(self, index: int) -> None:
        """Set the current frame index for side-by-side rendering."""
        new_index = max(0, min(index, self._get_frame_count() - 1))
        if new_index != self._index:
            self._index = new_index
            self.frameChanged.emit()
            self.frameIndexChanged.emit()

    @Slot()
    def nextFrame(self) -> None:
        """Advance to the next frame if available."""
        self.setFrame(self._index + 1)

    @Slot()
    def prevFrame(self) -> None:
        """Move to the previous frame if available."""
        self.setFrame(self._index - 1)

    # ------------------------------------------------------------------
    @Slot(str, str)
    def loadRuns(self, run_a: str, run_b: str) -> None:
        """Load metrics and frame paths from ``run_a`` and ``run_b``."""

        def read_metrics(path: Path) -> Dict[str, float]:
            totals: Dict[str, float] = defaultdict(float)
            file = path / "metrics.csv"
            if not file.exists():
                return totals
            with file.open() as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    try:
                        totals[row["category"]] += float(row["count"])
                    except (KeyError, ValueError):
                        continue
            return totals

        def collect_frames(path: Path) -> List[str]:
            frames = sorted((path / "frames").glob("*.png"))
            return [str(f) for f in frames]

        path_a = Path(run_a)
        path_b = Path(run_b)
        metrics_a = read_metrics(path_a)
        metrics_b = read_metrics(path_b)
        delta: Dict[str, float] = {}
        keys = set(metrics_a) | set(metrics_b)
        for k in keys:
            delta[k] = metrics_b.get(k, 0.0) - metrics_a.get(k, 0.0)
        self._metric_delta = delta

        self._frames_a = collect_frames(path_a)
        self._frames_b = collect_frames(path_b)
        self._index = 0

        self.metricDeltaChanged.emit()
        self.frameChanged.emit()
        self.frameCountChanged.emit()
        self.frameIndexChanged.emit()
