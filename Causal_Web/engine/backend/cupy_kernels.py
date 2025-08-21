"""CUDA kernels for heavy per-edge operations.

The functions in this module offload dense complex-number operations to the
GPU using `cupy` when ``Config.cupy_kernels`` is enabled. If `cupy` is not
installed or a CUDA device is unavailable, the implementations transparently
fall back to NumPy.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
from ...config import Config

try:  # pragma: no cover - optional dependency
    import cupy as cp
except Exception:  # pragma: no cover - gracefully handle missing CUDA
    cp = None


def _to_device(array: np.ndarray | Iterable[float]) -> "cp.ndarray" | np.ndarray:
    """Return an array on the GPU if `cupy` is available."""
    if cp is None:
        return np.asarray(array)
    return cp.asarray(array)


def complex_weighted_sum(phases: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Return ``phases * weights`` using the GPU when possible.

    Parameters
    ----------
    phases:
        Complex phase factors for each edge.
    weights:
        Edge weights with the same shape as ``phases``.

    Returns
    -------
    np.ndarray
        Weighted phases as a NumPy array regardless of backend.
    """

    if phases.shape != weights.shape:
        raise ValueError("phase and weight arrays must share a shape")

    if cp is None or not Config.cupy_kernels:
        return phases * weights

    c_phases = _to_device(phases)
    c_weights = _to_device(weights)
    result = c_phases * c_weights
    return cp.asnumpy(result)


def is_available() -> bool:
    """Return ``True`` when CuPy is selected and available."""

    return cp is not None and Config.backend == "cupy" and Config.cupy_kernels
