import numpy as np

from Causal_Web.engine.engine_v2.rho_delay import (
    update_rho_delay,
    update_rho_delay_vec,
)


def test_update_rho_delay():
    rho, d_eff = update_rho_delay(
        0.5,
        [0.5, 1.0],
        1.0,
        alpha_d=0.1,
        alpha_leak=0.05,
        eta=0.2,
        d0=3,
        gamma=2.0,
        rho0=0.1,
    )
    assert abs(rho - 0.7) < 1e-9
    assert d_eff == 7


def test_update_rho_delay_vec_intensity_array():
    rho_vec, d_vec = update_rho_delay_vec(
        np.array([0.5, 0.2], dtype=float),
        np.array([0.75, 0.25], dtype=float),
        np.array([1.0, 0.5], dtype=float),
        alpha_d=0.1,
        alpha_leak=0.05,
        eta=0.2,
        d0=np.array([3.0, 4.0], dtype=float),
        gamma=2.0,
        rho0=0.1,
    )
    r1, d1 = update_rho_delay(
        0.5,
        [0.75],
        1.0,
        alpha_d=0.1,
        alpha_leak=0.05,
        eta=0.2,
        d0=3.0,
        gamma=2.0,
        rho0=0.1,
    )
    r2, d2 = update_rho_delay(
        0.2,
        [0.25],
        0.5,
        alpha_d=0.1,
        alpha_leak=0.05,
        eta=0.2,
        d0=4.0,
        gamma=2.0,
        rho0=0.1,
    )
    assert np.allclose(rho_vec, [r1, r2])
    assert np.array_equal(d_vec, np.array([d1, d2]))
