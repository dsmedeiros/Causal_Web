from config.normalizer import Normalizer


def test_normalizer_roundtrip():
    norm = Normalizer()
    raw = {
        "Delta": 2.0,
        "W0": 1.0,
        "alpha_d": 10.0,
        "alpha_leak": 2.0,
        "sigma_reinforce": 4.0,
        "lambda_decay": 2.0,
        "a": 8.0,
        "b": 4.0,
        "eta": 0.5,
    }
    groups = norm.to_groups(raw)
    assert groups["Delta_over_W0"] == 2.0
    assert groups["alpha_d_over_leak"] == 5.0
    assert groups["sigma_reinforce_over_decay"] == 2.0
    assert groups["a_over_b"] == 2.0
    assert groups["eta_times_W0"] == 0.5
    base = {
        "Delta": 1.0,
        "W0": 1.0,
        "alpha_d": 1.0,
        "alpha_leak": 2.0,
        "sigma_reinforce": 1.0,
        "lambda_decay": 2.0,
        "a": 4.0,
        "b": 4.0,
        "eta": 0.1,
    }
    rebuilt = norm.to_raw(base, groups)
    assert rebuilt["Delta"] == 2.0
    assert rebuilt["alpha_d"] == 10.0
    assert rebuilt["sigma_reinforce"] == 4.0
    assert rebuilt["a"] == 8.0
    assert rebuilt["eta"] == 0.5
