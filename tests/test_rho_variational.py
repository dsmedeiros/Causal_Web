import math

from Causal_Web.engine.engine_v2.rho.variational import (
    lambda_to_coeffs,
    update_rho_variational,
    stamp_rho_metadata,
)


def test_lambda_to_coeffs():
    ad, al, et = lambda_to_coeffs(0.2, 0.1, 0.7, 0.5)
    assert math.isclose(ad, 0.2)
    assert math.isclose(al, 0.1)
    assert math.isclose(et, 0.35)


def test_update_rho_variational():
    rho, d_eff = update_rho_variational(
        1.0,
        [0.0, 2.0],
        3.0,
        lambda_s=0.2,
        lambda_l=0.1,
        lambda_I=0.7,
        eta=0.5,
        d0=1.0,
        gamma=1.0,
        rho0=1.0,
    )
    assert math.isclose(rho, 1.95)
    assert d_eff == 2


def test_stamp_rho_metadata():
    meta = {}
    stamp_rho_metadata(meta, "variational", {"lambda_s": 0.2})
    assert meta["rho_update"]["mode"] == "variational"
    assert meta["rho_update"]["lambda_s"] == 0.2
