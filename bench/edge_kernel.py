"""Benchmark complex multiplication throughput for edge kernel.

Runs 1e8 complex multiplications on CPU and, when available, on the GPU
using CuPy. Reports throughput in GB/s and relative speed-up.
"""

from __future__ import annotations

import time

import numpy as np

try:  # pragma: no cover - optional GPU path
    import cupy as cp

    _HAS_CUDA = cp.cuda.runtime.getDeviceCount() > 0
except Exception:  # pragma: no cover - CuPy not installed or no GPU
    cp = None
    _HAS_CUDA = False


BYTES_PER_OP = 48  # two inputs plus output at complex128
N = 1_000_000
REPEATS = 100  # N * REPEATS = 1e8 operations


def _bench_cpu() -> float:
    a = np.ones(N, dtype=np.complex128)
    b = np.ones(N, dtype=np.complex128)
    out = np.empty(N, dtype=np.complex128)
    start = time.perf_counter()
    for _ in range(REPEATS):
        np.multiply(a, b, out=out)
    end = time.perf_counter()
    elapsed = end - start
    return (N * REPEATS * BYTES_PER_OP) / elapsed / 1e9


def _bench_gpu() -> float:  # pragma: no cover - requires GPU
    a = cp.ones(N, dtype=cp.complex128)
    b = cp.ones(N, dtype=cp.complex128)
    out = cp.empty(N, dtype=cp.complex128)
    start = time.perf_counter()
    for _ in range(REPEATS):
        cp.multiply(a, b, out=out)
    cp.cuda.Stream.null.synchronize()
    end = time.perf_counter()
    elapsed = end - start
    return (N * REPEATS * BYTES_PER_OP) / elapsed / 1e9


def main() -> None:
    """Execute the benchmark and print throughput."""

    cpu_gbps = _bench_cpu()
    print(f"CPU: {cpu_gbps:.2f} GB/s")
    if _HAS_CUDA:
        gpu_gbps = _bench_gpu()
        print(f"GPU: {gpu_gbps:.2f} GB/s ({gpu_gbps / cpu_gbps:.1f}x)")
    else:  # pragma: no cover
        print("CuPy not available or no CUDA device detected.")


if __name__ == "__main__":  # pragma: no cover
    main()
