import pathlib

from invariants import checks


def test_causality_passes():
    deliveries = [{"d_arr": 1.0, "d_src": 0.5}]
    assert checks.causality(deliveries)


def test_causality_fails():
    deliveries = [{"d_arr": 0.2, "d_src": 0.5}]
    assert not checks.causality(deliveries)


def test_local_conservation():
    assert checks.local_conservation(1.0, 1.0 + 1e-7, 1e-6)


def test_no_signaling():
    assert checks.no_signaling(0.5, 1e-3)


def test_ancestry_determinism():
    seq = [("q1", "h1", "m1"), ("q2", "h2", "m2"), ("q1", "h1", "m1")]
    assert checks.ancestry_determinism(seq)
