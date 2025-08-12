from Causal_Web.engine.engine_v2.lccm import LCCM
from Causal_Web.engine.engine_v2.lccm.free_energy import (
    free_energy_score,
    stamp_lccm_metadata,
)

from Causal_Web.engine.engine_v2.lccm import LCCM
from Causal_Web.engine.engine_v2.lccm.free_energy import (
    free_energy_score,
    stamp_lccm_metadata,
)


def test_free_energy_score():
    score = free_energy_score(0.2, 0.6, 0.1, k_theta=1.0, k_c=1.0, k_q=0.5)
    assert score == 1.0 * (1 - 0.2) + 1.0 * 0.6 - 0.5 * 0.1


def test_lccm_free_energy_transition():
    l = LCCM(
        1,
        0.0,
        0.0,
        1.0,
        1.0,
        0.5,
        0.0,
        0.0,
        0.0,
        1.0,
        1,
        2,
        k_theta=1.0,
        k_c=1.0,
        mode="free_energy",
        k_q=0.0,
        F_min=0.3,
    )
    l.layer = "Θ"
    for _ in range(2):
        l.update_classical_metrics(0.0, 0.0, 1.0)
        l.update_eq(0.0)
        l.deliver()
    assert l.layer == "C"


def test_stamp_lccm_metadata():
    meta = {}
    stamp_lccm_metadata(meta, "free_energy", {"k_theta": 1.0})
    assert meta["lccm"]["mode"] == "free_energy"
    assert meta["lccm"]["k_theta"] == 1.0
