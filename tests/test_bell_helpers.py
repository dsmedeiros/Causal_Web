import numpy as np

from Causal_Web.engine.engine_v2.bell import Ancestry, BellHelpers


def test_setting_draw_modes() -> None:
    bell = BellHelpers(seed=123)
    anc = Ancestry()
    u, zeta = bell.lambda_at_source(anc, 0.5, 0.5)
    strict = bell.setting_draw("strict", anc, u, kappa_a=0)
    conditioned = bell.setting_draw("conditioned", anc, u, kappa_a=10.0)
    assert strict.shape == (3,)
    assert conditioned.shape == (3,)
    assert np.isclose(np.linalg.norm(strict), 1.0)
    assert np.isclose(np.linalg.norm(conditioned), 1.0)
    assert isinstance(zeta, float) and 0.0 <= zeta < 1.0


def test_contextual_readout_logging() -> None:
    bell = BellHelpers(seed=0)
    source = Ancestry()
    detector = Ancestry()
    u, zeta = bell.lambda_at_source(source, 0.5, 0.5)
    a_D = bell.setting_draw("strict", detector, u, kappa_a=0)
    outcome, log = bell.contextual_readout(
        "strict",
        a_D,
        detector,
        u,
        zeta,
        kappa_xi=1.0,
        source_ancestry=source,
        kappa_a=0.0,
        batch=1,
    )
    assert outcome in (-1, 1)
    assert log["mode"] == "strict"
    assert log["batch"] == 1
    assert 0.0 <= log["L"] <= 1.0
    assert isinstance(zeta, float) and 0.0 <= zeta < 1.0
