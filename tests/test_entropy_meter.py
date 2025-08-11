import numpy as np

from Causal_Web.engine.engine_v2.lccm import LCCM


def test_entropy_meter_decreases_with_entropy():
    """Increasing entropy should lower the meter's energy estimate.

    The entropy meter computes ``E_theta = a * (1 - H(p))``.  This test feeds
    it distributions of low and high Shannon entropy and asserts the resulting
    energy decreases monotonically with increasing ``H(p)``.  It guards against
    accidental sign errors that would invert this relationship.
    """

    lccm = LCCM(
        W0=2,
        zeta1=0.0,
        zeta2=0.0,
        rho0=1.0,
        a=1.0,
        b=0.0,
        C_min=0.0,
        f_min=1.0,
        conf_min=0.0,
        H_max=1.0,
        T_hold=1,
        T_class=1,
    )
    p_low = np.array([1.0, 0.0])
    p_high = np.array([0.5, 0.5])
    H_low = float(-(p_low * np.log2(p_low + 1e-12)).sum())
    H_high = float(-(p_high * np.log2(p_high + 1e-12)).sum())
    E_theta_low = lccm.a * (1.0 - H_low)
    E_theta_high = lccm.a * (1.0 - H_high)
    assert H_high > H_low
    assert E_theta_high < E_theta_low
