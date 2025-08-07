import numpy as np
from Causal_Web.engine.backend.cupy_kernels import complex_weighted_sum


def test_complex_weighted_sum_matches_scalar():
    n = 100_000
    rng = np.random.default_rng(0)
    phases = np.exp(1j * rng.random(n))
    weights = rng.random(n)
    expected = phases * weights
    result = complex_weighted_sum(phases, weights)
    np.testing.assert_allclose(result, expected, rtol=0.0, atol=1e-12)
