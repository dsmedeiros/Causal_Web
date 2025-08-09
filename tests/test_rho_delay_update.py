from Causal_Web.engine.engine_v2.rho_delay import update_rho_delay


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
