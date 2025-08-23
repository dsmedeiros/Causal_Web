"""Rolling telemetry buffers for live plots."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Mapping

import numpy as np


@dataclass
class RollingSeries:
    """Maintain a finite history of numeric samples."""

    maxlen: int
    _data: deque[float] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._data = deque(maxlen=self.maxlen)

    def append(self, value: float) -> None:
        """Append ``value`` keeping only the newest ``maxlen`` samples."""

        self._data.append(value)

    def as_list(self) -> list[float]:
        """Return the stored samples as a list."""

        return list(self._data)

    def bootstrap_ci(
        self,
        confidence: float = 0.95,
        n_boot: int = 1000,
        rng: np.random.Generator | None = None,
    ) -> tuple[float, float, float]:
        """Return mean and bootstrap confidence interval.

        Parameters
        ----------
        confidence:
            Two-sided confidence level. Defaults to ``0.95``.
        n_boot:
            Number of bootstrap resamples.
        rng:
            Optional NumPy random generator for deterministic resampling.

        Returns
        -------
        tuple
            ``(mean, lower, upper)`` of the estimated interval. ``NaN`` values
            are returned when the series is empty.
        """

        data = np.array(self._data, dtype=float)
        if data.size == 0:
            nan = float("nan")
            return nan, nan, nan
        rng = rng or np.random.default_rng()
        means = np.empty(n_boot, dtype=float)
        for i in range(n_boot):
            sample = rng.choice(data, size=data.size, replace=True)
            means[i] = float(np.mean(sample))
        alpha = 0.5 * (1.0 - confidence)
        lower = float(np.quantile(means, alpha))
        upper = float(np.quantile(means, 1.0 - alpha))
        return float(data.mean()), lower, upper

    def __len__(self) -> int:  # pragma: no cover - trivial
        """Return the number of stored samples."""

        return len(self._data)


@dataclass
class RollingTelemetry:
    """Store rolling histories for counters and invariants.

    Parameters
    ----------
    max_points:
        Maximum number of samples to retain per series.
    """

    max_points: int = 100
    counters: dict[str, RollingSeries] = field(default_factory=dict)
    invariants: dict[str, RollingSeries] = field(default_factory=dict)

    def record(
        self,
        counters: Mapping[str, float] | None = None,
        invariants: Mapping[str, bool] | None = None,
    ) -> None:
        """Record a new telemetry sample.

        ``counters`` and ``invariants`` are merged into their respective
        histories. Older samples are discarded once ``max_points`` is reached.
        """

        counters = counters or {}
        invariants = invariants or {}

        for key, value in counters.items():
            series = self.counters.setdefault(key, RollingSeries(self.max_points))
            series.append(float(value))
        for key, value in invariants.items():
            series = self.invariants.setdefault(key, RollingSeries(self.max_points))
            series.append(1.0 if value else 0.0)

    def get_counter_intervals(
        self,
        confidence: float = 0.95,
        n_boot: int = 1000,
        rng: np.random.Generator | None = None,
    ) -> dict[str, tuple[float, float, float]]:
        """Return bootstrap confidence intervals for all counters."""

        rng = rng or np.random.default_rng()
        out: dict[str, tuple[float, float, float]] = {}
        for key, series in self.counters.items():
            out[key] = series.bootstrap_ci(confidence, n_boot, rng)
        return out

    def get_counters(self) -> dict[str, list[float]]:
        """Return all counter histories as plain lists."""

        return {key: series.as_list() for key, series in self.counters.items()}

    def get_invariants(self) -> dict[str, list[float]]:
        """Return all invariant histories as plain lists."""

        return {key: series.as_list() for key, series in self.invariants.items()}
