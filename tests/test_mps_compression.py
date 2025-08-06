import numpy as np

from Causal_Web.engine.quantum.tensors import HADAMARD, propagate_chain


def test_hadamard_chain_memory_and_accuracy():
    unitaries = [HADAMARD] * 100
    psi = np.array([1 + 0j, 0 + 0j])
    final, mem = propagate_chain(unitaries, psi, chi_max=2)
    exact_amp = 2 ** (-50)
    err = abs(final[0] - exact_amp) / exact_amp
    assert mem < 50 * 1024 * 1024
    assert err < 0.01
