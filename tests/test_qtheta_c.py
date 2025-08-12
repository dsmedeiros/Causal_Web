import numpy as np
import pytest
from collections import deque

from Causal_Web.engine.engine_v2.qtheta_c import (
    deliver_packet,
    close_window,
    phase_stats,
    phase_stats_batch,
)


def test_deliver_packet_updates_fields():
    depth, psi_acc, p_v = 0, np.zeros(2, dtype=np.complex128), np.array([0.5, 0.5])
    bits = deque()
    packet = {"depth_arr": 2, "psi": [1.0, 0.0], "p": [0.2, 0.8], "bit": 1}
    edge = {
        "alpha": 0.5,
        "phi": 0.1,
        "A": 0.2,
        "U": [[1.0, 0.0], [0.0, 1.0]],
    }

    depth, psi_acc, p_v, (bit, conf), intensities, mu, kappa = deliver_packet(
        depth, psi_acc, p_v, bits, packet, edge
    )

    assert depth == 2
    assert np.allclose(p_v, np.array([0.4, 0.6]))
    assert bit == 1 and conf == 1.0
    assert intensities[0] == 1.0

    psi, EQ = close_window(psi_acc)
    assert np.isclose(EQ, np.vdot(psi_acc, psi_acc).real)
    if EQ > 0:
        assert np.allclose(psi, psi_acc / np.sqrt(EQ))


def test_intensity_theta_layer_ignores_other_contributions():
    depth, psi_acc, p_v = 0, np.zeros(2, dtype=np.complex128), np.array([0.5, 0.5])
    bits = deque()
    packet = {"depth_arr": 1, "psi": [1.0, 0.0], "p": [0.1, 0.2], "bit": 1}
    edge = {"alpha": 1.0, "phi": 0.0, "A": 0.0, "U": [[1.0, 0.0], [0.0, 1.0]]}

    _, _, _, _, intensities, mu, kappa = deliver_packet(
        depth, psi_acc, p_v, bits, packet, edge
    )
    assert intensities[1] == pytest.approx(0.3, abs=1e-6)


def test_intensity_c_layer_only_counts_bits():
    depth, psi_acc, p_v = 0, np.zeros(2, dtype=np.complex128), np.array([0.5, 0.5])
    bits = deque()
    packet = {"depth_arr": 1, "psi": [1.0, 0.0], "p": [0.1, 0.2], "bit": 0}
    edge = {"alpha": 1.0, "phi": 0.0, "A": 0.0, "U": [[1.0, 0.0], [0.0, 1.0]]}

    _, _, _, _, intensities, mu, kappa = deliver_packet(
        depth, psi_acc, p_v, bits, packet, edge
    )
    assert intensities[2] == 0.0


def test_p_v_unchanged_when_update_p_false():
    depth, psi_acc, p_v = 0, np.zeros(2, dtype=np.complex128), np.array([0.7, 0.3])
    bits = deque()
    packet = {"depth_arr": 1, "psi": [1.0, 0.0], "p": [0.1, 0.9], "bit": 1}
    edge = {"alpha": 1.0, "phi": 0.0, "A": 0.0, "U": [[1.0, 0.0], [0.0, 1.0]]}

    _, _, p_out, _, _, _, _ = deliver_packet(
        depth, psi_acc, p_v, bits, packet, edge, update_p=False
    )
    assert np.allclose(p_out, p_v)


def test_phase_stats_batch_matches_single():
    U = [
        np.eye(2, dtype=np.complex64),
        np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex64),
    ]
    phase = [1.0 + 0.0j, np.exp(1j * 0.5)]
    psi = [
        np.array([1.0, 0.0], dtype=np.complex64),
        np.array([1 / np.sqrt(2), 1 / np.sqrt(2)], dtype=np.complex64),
    ]
    mu_single, kappa_single = zip(
        *(
            phase_stats(U_i, phase_i, psi_i)[:2]
            for U_i, phase_i, psi_i in zip(U, phase, psi)
        )
    )
    mu_batch, kappa_batch, _ = phase_stats_batch(U, phase, psi)
    assert np.allclose(mu_batch, mu_single)
    assert np.allclose(kappa_batch, kappa_single)
