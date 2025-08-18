import numpy as np
import pytest

cupy = pytest.importorskip("cupy")
from Causal_Web.engine.backend.cupy_kernels import complex_weighted_sum


def test_complex_weighted_sum_matches_scalar():
    try:
        if cupy.cuda.runtime.getDeviceCount() == 0:
            pytest.skip("No CUDA device")
    except cupy.cuda.runtime.CUDARuntimeError:
        pytest.skip("CUDA device unavailable")

    n = 100_000
    rng = np.random.default_rng(0)
    phases = np.exp(1j * rng.random(n))
    weights = rng.random(n)
    expected = phases * weights
    result = complex_weighted_sum(phases, weights)
    np.testing.assert_allclose(result, expected, rtol=0.0, atol=1e-12)
