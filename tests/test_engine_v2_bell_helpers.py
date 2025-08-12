import numpy as np

from Causal_Web.engine.engine_v2.bell import Ancestry, BellHelpers


def test_bell_helpers_return_unit_vectors() -> None:
    helper = BellHelpers(seed=0)
    anc = Ancestry()
    lam_u, _ = helper.lambda_at_source(anc, 0.5, 0.5)
    assert np.isclose(np.linalg.norm(lam_u), 1.0)
    setting = helper.setting_draw("strict", anc, lam_u, 0.0)
    assert np.isclose(np.linalg.norm(setting), 1.0)
