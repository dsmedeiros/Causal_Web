import numpy as np
from collections import deque

from Causal_Web.engine.engine_v2.qtheta_c import deliver_packet, close_window


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

    depth, psi_acc, p_v, (bit, conf), intensity = deliver_packet(
        depth, psi_acc, p_v, bits, packet, edge
    )

    assert depth == 2
    assert np.allclose(p_v, np.array([0.4, 0.6]))
    assert bit == 1 and conf == 1.0
    assert intensity == 1.0

    psi, EQ = close_window(psi_acc)
    assert np.isclose(EQ, np.vdot(psi_acc, psi_acc).real)
    if EQ > 0:
        assert np.allclose(psi, psi_acc / np.sqrt(EQ))
