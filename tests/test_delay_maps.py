from Causal_Web.delay_maps import PhiLinear, log_scalar, stamp_delay_metadata


def test_log_scalar_mapping():
    out = log_scalar(2.0, d0=1.0, gamma=2.0, rho0=1.0)
    assert out == 3


def test_phi_linear_stronger_delay():
    mapper = PhiLinear(alpha=0.1)
    mapper.phi[2] = 5.0
    mapper.update_vertex(1, [2], rho_bar=0.0)
    d_phi = mapper.effective_delay(5.0, 1, 2)
    d_log = log_scalar(0.0, d0=5.0, gamma=1.0, rho0=1.0)
    assert d_phi > d_log


def test_stamp_delay_metadata():
    meta = {}
    stamp_delay_metadata(meta, "phi_linear", {"alpha": 0.05})
    assert meta["delay_map"]["mode"] == "phi_linear"
    assert meta["delay_map"]["alpha"] == 0.05
