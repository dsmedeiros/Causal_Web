from __future__ import annotations

import numpy as np

from Causal_Web.engine.engine_v2.bell import (
    Ancestry,
    BellHelpers,
    _splitmix64,
)


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


def test_lambda_determinism() -> None:
    anc = Ancestry()
    bell1 = BellHelpers(seed=42)
    bell2 = BellHelpers(seed=42)
    u1, z1 = bell1.lambda_at_source(anc, 0.5, 1.0)
    u2, z2 = bell2.lambda_at_source(anc, 0.5, 1.0)
    assert np.allclose(u1, u2)
    assert z1 == z2
    det = Ancestry()
    a1 = bell1.setting_draw("strict", det, u1, kappa_a=0.0)
    a2 = bell2.setting_draw("strict", det, u2, kappa_a=0.0)
    o1, _ = bell1.contextual_readout(
        "strict",
        a1,
        det,
        u1,
        z1,
        kappa_xi=1.0,
        source_ancestry=anc,
        kappa_a=0.0,
        batch=0,
    )
    o2, _ = bell2.contextual_readout(
        "strict",
        a2,
        det,
        u2,
        z2,
        kappa_xi=1.0,
        source_ancestry=anc,
        kappa_a=0.0,
        batch=0,
    )
    assert o1 == o2


def test_zeta_uniformity() -> None:
    bell = BellHelpers(seed=0)
    zetas = []
    for i in range(1000):
        anc = Ancestry(h=np.array([np.uint64(i), 0, 0, 0], dtype=np.uint64))
        _, z = bell.lambda_at_source(anc, 0.0, 1.0)
        zetas.append(z)
    zetas = np.array(zetas)
    assert abs(np.mean(zetas) - 0.5) < 0.01
    assert abs(np.var(zetas) - 1.0 / 12.0) < 0.01


def _simulate_chsh(
    mode: str, kappa_a: float, kappa_xi: float, trials: int = 2000
) -> tuple[float, float, float]:
    bell = BellHelpers(seed=0)
    src = Ancestry()
    detA = Ancestry()
    detB = Ancestry(h=np.array([np.uint64(1), 0, 0, 0], dtype=np.uint64))
    combos = {(0, 0): [], (0, 1): [], (1, 0): [], (1, 1): []}
    margA: list[int] = []
    margB: list[int] = []
    for _ in range(trials):
        u, z = bell.lambda_at_source(src, 0.0, 1.0)
        a_setting = bell._rng.integers(2)
        b_setting = bell._rng.integers(2)
        if a_setting == 0:
            a_vec = bell.setting_draw("strict", detA, u, 0.0)
        else:
            a_vec = bell.setting_draw(mode, detA, u, kappa_a)
        if b_setting == 0:
            b_vec = bell.setting_draw("strict", detB, u, 0.0)
        else:
            b_vec = bell.setting_draw(mode, detB, u, kappa_a)
        oA, _ = bell.contextual_readout(
            mode,
            a_vec,
            detA,
            u,
            z,
            kappa_xi,
            source_ancestry=src,
            kappa_a=kappa_a,
            batch=0,
        )
        oB, _ = bell.contextual_readout(
            mode,
            b_vec,
            detB,
            u,
            z,
            kappa_xi,
            source_ancestry=src,
            kappa_a=kappa_a,
            batch=0,
        )
        combos[(a_setting, b_setting)].append(oA * oB)
        margA.append(oA)
        margB.append(oB)
    E = {k: np.mean(v) for k, v in combos.items()}
    s_val = abs(E[(0, 0)] + E[(0, 1)] + E[(1, 0)] - E[(1, 1)])
    pA = (np.array(margA) > 0).mean()
    pB = (np.array(margB) > 0).mean()
    return s_val, pA, pB


def test_no_signaling() -> None:
    s_strict, pA_s, pB_s = _simulate_chsh("strict", 0.0, 0.1)
    assert s_strict <= 2.0 + 0.1
    assert abs(pA_s - 0.5) < 0.02
    assert abs(pB_s - 0.5) < 0.02

    s_cond, pA_c, pB_c = _simulate_chsh("conditioned", 10.0, 0.1)
    assert s_cond > s_strict
    assert abs(pA_c - 0.5) < 0.02
    assert abs(pB_c - 0.5) < 0.02


def test_zeta_platform_stability() -> None:
    bell = BellHelpers(seed=0)
    anc = Ancestry(h=np.array([np.uint64(0), 0, 0, 0], dtype=np.uint64))
    _, z = bell.lambda_at_source(anc, 0.0, 1.0)
    expected = np.float64(_splitmix64(np.uint64(0))) / np.float64(2**64)
    assert z == expected
