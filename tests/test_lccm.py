from Causal_Web.engine.engine_v2.lccm import LCCM


def test_lccm_layer_transitions_with_hysteresis():
    lccm = LCCM(
        W0=2,
        zeta1=0.0,
        zeta2=0.0,
        rho0=1.0,
        a=1.0,
        b=0.5,
        C_min=1.0,
        f_min=0.9,
        conf_min=0.9,
        H_max=0.5,
        T_hold=2,
        T_class=2,
        deg=0,
        rho_mean=0.0,
    )

    # Q -> Θ when λ >= a*W
    lccm.advance_depth(0)
    lccm.deliver(True)
    assert lccm.layer == "Q"
    lccm.deliver(True)
    assert lccm.layer == "Θ"

    # Θ -> Q requires λ <= b*W and EQ above threshold for T_hold
    lccm.advance_depth(2)
    lccm.update_eq(1.0)
    lccm.deliver(False)
    assert lccm.layer == "Θ"
    lccm.advance_depth(4)
    lccm.update_eq(1.0)
    lccm.deliver(False)
    assert lccm.layer == "Q"

    # Decoherence again for Θ -> C test
    lccm.advance_depth(5)
    lccm.deliver(True)
    lccm.deliver(True)
    assert lccm.layer == "Θ"

    # Θ -> C when classical dominance sustained for T_class
    lccm.update_classical_metrics(1.0, 0.0, 1.0)
    lccm.deliver(False)
    assert lccm.layer == "Θ"
    lccm.advance_depth(6)
    lccm.update_classical_metrics(1.0, 0.0, 1.0)
    lccm.deliver(False)
    assert lccm.layer == "C"


def test_window_size_grows_with_density():
    lccm = LCCM(
        W0=2,
        zeta1=0.0,
        zeta2=1.0,
        rho0=1.0,
        a=1.0,
        b=0.5,
        C_min=1.0,
        f_min=0.0,
        conf_min=0.0,
        H_max=1.0,
        T_hold=1,
        T_class=1,
        deg=1,
        rho_mean=0.0,
    )

    w0 = lccm._window_size()
    lccm.rho_mean = 10.0
    w1 = lccm._window_size()
    assert w1 > w0


def test_theta_to_c_requires_sustained_confidence():
    lccm = LCCM(
        W0=2,
        zeta1=0.0,
        zeta2=0.0,
        rho0=1.0,
        a=1.0,
        b=0.5,
        C_min=1.0,
        f_min=0.5,
        conf_min=0.8,
        H_max=0.5,
        T_hold=1,
        T_class=3,
        deg=0,
        rho_mean=0.0,
    )

    lccm.layer = "Θ"

    for _ in range(2):
        lccm.update_classical_metrics(0.6, 0.1, 0.9)
        lccm.deliver(False)
        assert lccm.layer == "Θ"

    lccm.update_classical_metrics(0.6, 0.1, 0.7)
    lccm.deliver(False)
    assert lccm.layer == "Θ"

    for _ in range(3):
        lccm.update_classical_metrics(0.6, 0.1, 0.9)
        lccm.deliver(False)

    assert lccm.layer == "C"
