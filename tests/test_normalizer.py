from config.normalizer import Normalizer


def test_normalizer_roundtrip():
    norm = Normalizer()
    raw = {"Delta": 2.0, "W0": 1.0, "alpha_d": 10.0, "alpha_leak": 2.0}
    groups = norm.to_groups(raw)
    assert groups["Delta_over_W0"] == 2.0
    assert groups["alpha_d_over_leak"] == 5.0
    base = {"Delta": 1.0, "W0": 1.0, "alpha_d": 1.0, "alpha_leak": 2.0}
    rebuilt = norm.to_raw(base, groups)
    assert rebuilt["Delta"] == 2.0
    assert rebuilt["alpha_d"] == 10.0
