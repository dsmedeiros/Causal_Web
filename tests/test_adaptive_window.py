import math

from Causal_Web.engine.engine_v2.lccm import WindowParams, WindowState, on_window_close


def test_on_window_close_updates_state():
    params = WindowParams()
    state = WindowState(M_v=0.0, W_v=8.0)
    rhos = [0.5, 1.0, 1.5]
    weights = [1.0, 1.0, 1.0]
    on_window_close(rhos, weights, params, state)

    assert 0 < state.M_v < 1.0
    assert state.W_v > 8.0


def test_rate_limited_update_mu():
    params = WindowParams(beta=None, mu=0.1)
    state = WindowState(M_v=0.0, W_v=10.0)
    rhos = [10.0, 10.0, 10.0]
    weights = [1.0, 1.0, 1.0]
    on_window_close(rhos, weights, params, state)
    assert 10.0 < state.W_v <= 11.0


def test_robust_mean_trims_outliers():
    params = WindowParams()
    state = WindowState()
    rhos = [1.0, 1.0, 100.0]
    weights = [0.45, 0.45, 0.10]
    on_window_close(rhos, weights, params, state)
    assert state.M_v < 5.0
