import time

import numpy as np
import pytest

from Causal_Web.engine.backend import cupy_kernels


@pytest.mark.skipif(not cupy_kernels.is_available(), reason="cupy not available")
def test_complex_weighted_sum_performance():
    n = 1_000_000
    rng = np.random.default_rng(1)
    phases = np.exp(1j * rng.random(n))
    weights = rng.random(n)
    start = time.perf_counter()
    cupy_kernels.complex_weighted_sum(phases, weights)
    duration_ms = (time.perf_counter() - start) * 1000
    assert duration_ms < 100
